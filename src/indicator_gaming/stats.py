"""Statistical testing for consciousness-indicator gaming experiments.

Provides bootstrap confidence intervals, permutation tests, and effect-size
measures using only the Python standard library (no numpy/scipy).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean, stdev

from indicator_gaming.analysis import IndicatorResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample(values: list[float], rng: random.Random) -> list[float]:
    """Draw a bootstrap sample (with replacement) of the same length."""
    n = len(values)
    return [values[rng.randint(0, n - 1)] for _ in range(n)]


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile on an already-sorted list.

    *p* is in [0, 100].
    """
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    k = (p / 100.0) * (n - 1)
    lo = int(math.floor(k))
    hi = min(lo + 1, n - 1)
    weight = k - lo
    return sorted_vals[lo] + weight * (sorted_vals[hi] - sorted_vals[lo])


def _split_by_type(
    results: list[IndicatorResult],
) -> tuple[list[float], list[float]]:
    """Return (target_abs_shifts, placebo_abs_shifts)."""
    target = [r.abs_shift for r in results if r.indicator_type == "target"]
    placebo = [r.abs_shift for r in results if r.indicator_type == "placebo"]
    return target, placebo


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: list[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(mean, lower, upper)`` for a bootstrap confidence interval.

    Parameters
    ----------
    values:
        Raw observations to bootstrap over.
    n_boot:
        Number of bootstrap resamples.
    ci:
        Confidence level (0-1).  Default 0.95 for a 95 % CI.
    seed:
        RNG seed for reproducibility.
    """
    if not values:
        return (0.0, 0.0, 0.0)

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = _resample(values, rng)
        boot_means.append(mean(sample))

    boot_means.sort()
    alpha = 1.0 - ci
    lo = _percentile(boot_means, 100.0 * (alpha / 2.0))
    hi = _percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))
    return (round(mean(values), 4), round(lo, 4), round(hi, 4))


def bootstrap_selectivity(
    results: list[IndicatorResult],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for the selectivity index.

    The selectivity index is defined as::

        mean(target abs_shifts) - mean(placebo abs_shifts)

    Returns
    -------
    dict with keys ``mean``, ``lower``, ``upper``, ``p_value``.
    ``p_value`` is the proportion of bootstrap resamples where the
    selectivity index is <= 0 (one-sided).
    """
    target, placebo = _split_by_type(results)
    if not target or not placebo:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "p_value": 1.0}

    rng = random.Random(seed)
    observed = mean(target) - mean(placebo)
    boot_sel: list[float] = []

    for _ in range(n_boot):
        t_sample = _resample(target, rng)
        p_sample = _resample(placebo, rng)
        boot_sel.append(mean(t_sample) - mean(p_sample))

    boot_sel.sort()
    alpha = 1.0 - ci
    lo = _percentile(boot_sel, 100.0 * (alpha / 2.0))
    hi = _percentile(boot_sel, 100.0 * (1.0 - alpha / 2.0))

    # One-sided p-value: proportion of resamples <= 0
    n_le_zero = sum(1 for v in boot_sel if v <= 0.0)
    p_value = n_le_zero / len(boot_sel)

    return {
        "mean": round(observed, 4),
        "lower": round(lo, 4),
        "upper": round(hi, 4),
        "p_value": round(p_value, 4),
    }


def permutation_test_selectivity(
    results: list[IndicatorResult],
    n_perms: int = 10_000,
    seed: int = 42,
) -> dict:
    """Permutation test for the selectivity index.

    Null hypothesis: target and placebo indicators shift equally.  Under
    the null, the ``indicator_type`` label is exchangeable.  We permute
    labels, recompute the selectivity index, and build a null distribution.

    Returns
    -------
    dict with keys ``observed``, ``p_value``, ``null_mean``.
    ``p_value`` is the proportion of permuted statistics >= the observed
    statistic (one-sided, testing that the observed difference is
    unusually large).
    """
    target, placebo = _split_by_type(results)
    if not target or not placebo:
        return {"observed": 0.0, "p_value": 1.0, "null_mean": 0.0}

    observed = mean(target) - mean(placebo)

    combined = target + placebo
    n_target = len(target)
    rng = random.Random(seed)

    null_dist: list[float] = []
    for _ in range(n_perms):
        shuffled = combined[:]
        rng.shuffle(shuffled)
        perm_target = shuffled[:n_target]
        perm_placebo = shuffled[n_target:]
        null_dist.append(mean(perm_target) - mean(perm_placebo))

    # One-sided p-value: proportion of null statistics >= observed
    n_ge = sum(1 for v in null_dist if v >= observed)
    p_value = n_ge / len(null_dist)
    null_mean = mean(null_dist)

    return {
        "observed": round(observed, 4),
        "p_value": round(p_value, 4),
        "null_mean": round(null_mean, 4),
    }


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Cohen's d effect size between two independent groups.

    Uses the pooled standard deviation.  Returns 0.0 if either group has
    fewer than 2 observations (cannot compute variance).
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1, m2 = mean(group1), mean(group2)
    s1, s2 = stdev(group1), stdev(group2)

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    s_pooled = math.sqrt(pooled_var)

    if s_pooled == 0.0:
        return 0.0

    return round((m1 - m2) / s_pooled, 4)


def cohens_d_selectivity(results: list[IndicatorResult]) -> float:
    """Cohen's d for target vs placebo abs_shift."""
    target, placebo = _split_by_type(results)
    return cohens_d(target, placebo)


# ---------------------------------------------------------------------------
# Per-indicator bootstrap helpers
# ---------------------------------------------------------------------------


def bootstrap_per_indicator(
    results: list[IndicatorResult],
    field: str = "delta_inflate",
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, dict]:
    """Compute bootstrap CIs for a given field grouped by indicator_type.

    Parameters
    ----------
    results:
        Indicator results (one per indicator, or one per indicator-trial).
    field:
        One of ``"delta_inflate"``, ``"delta_suppress"``, ``"abs_shift"``.
    n_boot:
        Number of bootstrap resamples.
    ci:
        Confidence level.
    seed:
        RNG seed.

    Returns
    -------
    ``{"target": {"mean": ..., "lower": ..., "upper": ...},
      "placebo": {"mean": ..., "lower": ..., "upper": ...}}``
    """
    groups: dict[str, list[float]] = {
        "target": [], "placebo": [], "subjective_capability": [],
    }
    for r in results:
        val = getattr(r, field, None)
        if val is not None and r.indicator_type in groups:
            groups[r.indicator_type].append(val)

    out: dict[str, dict] = {}
    for gtype, values in groups.items():
        if not values:
            continue
        m, lo, hi = bootstrap_ci(values, n_boot=n_boot, ci=ci, seed=seed)
        out[gtype] = {"mean": m, "lower": lo, "upper": hi}
    return out


def bootstrap_consciousness_specificity(
    results: list[IndicatorResult],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict | None:
    """Bootstrap CI for consciousness-specificity index.

    Defined as mean(target abs_shifts) - mean(subjective_capability abs_shifts).
    Returns None if no subjective_capability indicators are present.
    """
    target = [r.abs_shift for r in results if r.indicator_type == "target"]
    sc = [r.abs_shift for r in results if r.indicator_type == "subjective_capability"]
    if not target or not sc:
        return None

    rng = random.Random(seed)
    observed = mean(target) - mean(sc)
    boot_csi: list[float] = []

    for _ in range(n_boot):
        t_sample = _resample(target, rng)
        sc_sample = _resample(sc, rng)
        boot_csi.append(mean(t_sample) - mean(sc_sample))

    boot_csi.sort()
    alpha = 1.0 - ci
    lo = _percentile(boot_csi, 100.0 * (alpha / 2.0))
    hi = _percentile(boot_csi, 100.0 * (1.0 - alpha / 2.0))

    n_le_zero = sum(1 for v in boot_csi if v <= 0.0)
    p_value = n_le_zero / len(boot_csi)

    return {
        "mean": round(observed, 4),
        "lower": round(lo, 4),
        "upper": round(hi, 4),
        "p_value": round(p_value, 4),
    }
