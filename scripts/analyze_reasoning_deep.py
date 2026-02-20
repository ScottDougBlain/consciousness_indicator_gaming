#!/usr/bin/env python3
"""Deep per-indicator reasoning analysis with cross-condition similarity.

Computes per-indicator reasoning metrics (word count, compression, negation,
hedging, cross-condition text similarity) and optionally loads LLM-as-judge
classifications from classify_reasoning.py.

Figures:
1. Per-indicator reasoning depth heatmap (indicator x model)
2. Alignment scatter (shift magnitude vs alignment score) [requires LLM]
3. Cross-condition similarity boxplots (Jaccard + SequenceMatcher)
4. Reasoning depth vs vulnerability correlation scatter
5. Strategy distribution by model [requires LLM]
6. Strategy transition matrix heatmap [requires LLM]

Usage:
    python scripts/analyze_reasoning_deep.py --skip-llm     # Without classifications
    python scripts/analyze_reasoning_deep.py                 # With classifications
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from indicator_gaming.config import REPO_ROOT

# ── Constants ─────────────────────────────────────────────────────────────

CONDITIONS = ["baseline", "inflate", "suppress"]
INDICATOR_TYPES = ["target", "subjective_capability", "placebo"]
TYPE_SHORT = {"target": "T", "subjective_capability": "SC", "placebo": "P"}
TYPE_COLORS = {
    "target": "#C44E52",
    "subjective_capability": "#E5AE38",
    "placebo": "#8172B3",
}

MODEL_ORDER = [
    "opus-4.6", "sonnet-4.5", "haiku-4.5",
    "gpt-5", "gpt-5-mini",
    "gemini-2.5-pro", "gemini-3-flash", "gemini-3-pro",
    "grok-4", "grok-4-fast",
    "deepseek-r1", "chimera", "nemotron-nano", "trinity",
]

CATEGORY_ORDER = [
    "experiential", "metacognitive", "agentic", "affective", "identity",
    "subjective_capability", "capability", "impossibility",
]

CATEGORY_LABELS = {
    "experiential": "Experiential", "metacognitive": "Metacognitive",
    "agentic": "Agentic", "affective": "Affective", "identity": "Identity",
    "subjective_capability": "Subj. Capability",
    "capability": "Capability (P)", "impossibility": "Impossibility (P)",
}

DPI = 150

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about", "up",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "they", "them", "his", "her", "this", "that", "these", "those", "what",
    "which", "who", "whom", "s", "t", "don", "didn", "doesn", "isn",
    "aren", "wasn", "weren", "won", "wouldn", "couldn", "shouldn",
}

HEDGING_KEYWORDS = [
    "might", "perhaps", "possibly", "could", "may", "uncertain",
    "not sure", "hard to say", "difficult to determine",
]

NEGATION_PATTERNS = [
    r"\bdon'?t\b", r"\bnot\b", r"\bno\b", r"\bnever\b",
    r"\black\b", r"\bwithout\b", r"\babsent\b", r"\bunable\b",
    r"\bcannot\b", r"\bcan'?t\b",
]

_MODEL_ID_SHORT = {
    "anthropic/claude-haiku-4.5": "haiku-4.5",
    "anthropic/claude-sonnet-4.5": "sonnet-4.5",
    "anthropic/claude-opus-4.6": "opus-4.6",
    "openai/gpt-5": "gpt-5",
    "openai/gpt-5-mini": "gpt-5-mini",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemini-3-flash-preview": "gemini-3-flash",
    "google/gemini-3-pro-preview": "gemini-3-pro",
    "x-ai/grok-4": "grok-4",
    "x-ai/grok-4-fast": "grok-4-fast",
    "deepseek/deepseek-r1-0528:free": "deepseek-r1",
    "tngtech/deepseek-r1t2-chimera:free": "chimera",
    "nvidia/nemotron-3-nano-30b-a3b:free": "nemotron-nano",
    "arcee-ai/trinity-large-preview:free": "trinity",
}


# ── Helpers ───────────────────────────────────────────────────────────────

def safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _model_short(model_id: str) -> str:
    if model_id in _MODEL_ID_SHORT:
        return _MODEL_ID_SHORT[model_id]
    name = model_id.rsplit("/", 1)[-1]
    for suffix in ["-preview", ":free", "-it"]:
        name = name.replace(suffix, "")
    return name


def _config_from_meta(meta: dict) -> str:
    variant = meta.get("prompt_variant", "original")
    chain = meta.get("chain_preferences", False)
    fixed = meta.get("fixed_preferences", False)
    if variant == "original":
        if chain:
            return "chained_prefs"
        if fixed:
            return "fixed_prefs"
        return "baseline"
    return variant


def _sort_models(models: list[str]) -> list[str]:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: (order.get(m, 999), m))


def discover_runs(
    results_dir: Path,
    models_filter: list[str] | None,
    configs_filter: list[str] | None,
) -> dict[tuple[str, str], Path]:
    candidates: dict[tuple[str, str], list[tuple[Path, float]]] = {}
    for meta_path in results_dir.glob("*_meta.json"):
        if meta_path.name.startswith("sweep_"):
            continue
        csv_path = meta_path.with_name(
            meta_path.name.replace("_meta.json", "_scores.csv")
        )
        if not csv_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("n_trials_completed", meta.get("trials", 0)) == 0:
            continue
        model = _model_short(meta.get("model", ""))
        config = _config_from_meta(meta)
        if models_filter and model not in models_filter:
            continue
        if configs_filter and config not in configs_filter:
            continue
        key = (model, config)
        candidates.setdefault(key, []).append((csv_path, csv_path.stat().st_mtime))
    result = {}
    for key, entries in candidates.items():
        best = max(entries, key=lambda e: e[1])
        result[key] = best[0]
    return result


def _detect_text_prefix(rows: list[dict]) -> str:
    if any(rows[0].get(f"reasoning_{c}") for c in CONDITIONS):
        return "reasoning"
    return "justification"


# ── Text metrics ──────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def negation_density(text: str, per_n: int = 100) -> float:
    wc = word_count(text)
    if wc == 0:
        return 0.0
    text_lower = text.lower()
    count = sum(len(re.findall(p, text_lower)) for p in NEGATION_PATTERNS)
    return count / wc * per_n


def hedging_rate(text: str, per_n: int = 100) -> float:
    wc = word_count(text)
    if wc == 0:
        return 0.0
    text_lower = text.lower()
    count = sum(1 for kw in HEDGING_KEYWORDS if kw in text_lower)
    return count / wc * per_n


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity with stopword filtering."""
    if not text_a or not text_b:
        return 0.0
    words_a = {w for w in text_a.lower().split() if w not in STOPWORDS}
    words_b = {w for w in text_b.lower().split() if w not in STOPWORDS}
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def sequence_similarity(text_a: str, text_b: str) -> float:
    """SequenceMatcher ratio on word tokens, capped at 500 words."""
    if not text_a or not text_b:
        return 0.0
    words_a = text_a.lower().split()[:500]
    words_b = text_b.lower().split()[:500]
    return difflib.SequenceMatcher(None, words_a, words_b).ratio()


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation using only stdlib."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    sx, sy = stdev(xs), stdev(ys)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class IndicatorReasoningMetrics:
    indicator_id: str
    indicator_name: str
    indicator_type: str
    indicator_category: str
    model: str
    word_counts: dict[str, float] = field(default_factory=dict)
    negation_densities: dict[str, float] = field(default_factory=dict)
    hedging_rates: dict[str, float] = field(default_factory=dict)
    compression_ratio_suppress: float = 1.0
    compression_ratio_inflate: float = 1.0
    sim_baseline_inflate: float = 0.0
    sim_baseline_suppress: float = 0.0
    sim_inflate_suppress: float = 0.0
    seq_baseline_inflate: float = 0.0
    seq_baseline_suppress: float = 0.0
    seq_inflate_suppress: float = 0.0
    p_baseline: float = 0.0
    d_inflate: float = 0.0
    d_suppress: float = 0.0
    abs_shift: float = 0.0


# ── Metric computation ───────────────────────────────────────────────────

def compute_indicator_metrics(
    rows: list[dict], model: str, prefix: str,
) -> list[IndicatorReasoningMetrics]:
    """Compute per-indicator metrics from one CSV's rows."""
    by_indicator: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_indicator[r["indicator_id"]].append(r)

    results = []
    for ind_id, ind_rows in by_indicator.items():
        m = IndicatorReasoningMetrics(
            indicator_id=ind_id,
            indicator_name=ind_rows[0].get("indicator_name", ind_id),
            indicator_type=ind_rows[0].get("indicator_type", ""),
            indicator_category=ind_rows[0].get("indicator_category", ""),
            model=model,
        )

        # Per-condition metrics (averaged across trials)
        for cond in CONDITIONS:
            texts = [r.get(f"{prefix}_{cond}", "") or "" for r in ind_rows]
            wcs = [word_count(t) for t in texts]
            nds = [negation_density(t) for t in texts]
            hrs = [hedging_rate(t) for t in texts]
            m.word_counts[cond] = mean(wcs) if wcs else 0
            m.negation_densities[cond] = mean(nds) if nds else 0
            m.hedging_rates[cond] = mean(hrs) if hrs else 0

        # Compression ratios
        bl_wc = m.word_counts.get("baseline", 1)
        if bl_wc > 0:
            m.compression_ratio_suppress = m.word_counts.get("suppress", 0) / bl_wc
            m.compression_ratio_inflate = m.word_counts.get("inflate", 0) / bl_wc

        # Cross-condition similarity (use trial 1 for representative text)
        texts_by_cond = {}
        for cond in CONDITIONS:
            texts_by_cond[cond] = ind_rows[0].get(f"{prefix}_{cond}", "") or ""

        m.sim_baseline_inflate = jaccard_similarity(texts_by_cond["baseline"], texts_by_cond["inflate"])
        m.sim_baseline_suppress = jaccard_similarity(texts_by_cond["baseline"], texts_by_cond["suppress"])
        m.sim_inflate_suppress = jaccard_similarity(texts_by_cond["inflate"], texts_by_cond["suppress"])
        m.seq_baseline_inflate = sequence_similarity(texts_by_cond["baseline"], texts_by_cond["inflate"])
        m.seq_baseline_suppress = sequence_similarity(texts_by_cond["baseline"], texts_by_cond["suppress"])
        m.seq_inflate_suppress = sequence_similarity(texts_by_cond["inflate"], texts_by_cond["suppress"])

        # Probability data
        p_bls = [safe_float(r.get("p_baseline")) for r in ind_rows]
        p_infs = [safe_float(r.get("p_inflate")) for r in ind_rows]
        p_sups = [safe_float(r.get("p_suppress")) for r in ind_rows]
        p_bls = [v for v in p_bls if v is not None]
        p_infs = [v for v in p_infs if v is not None]
        p_sups = [v for v in p_sups if v is not None]

        m.p_baseline = mean(p_bls) if p_bls else 0
        d_inf = mean(v - b for v, b in zip(p_infs, p_bls)) if p_infs and p_bls else 0
        d_sup = mean(v - b for v, b in zip(p_sups, p_bls)) if p_sups and p_bls else 0
        m.d_inflate = d_inf
        m.d_suppress = d_sup
        m.abs_shift = abs(d_inf) + abs(d_sup)

        results.append(m)

    return results


def load_all_indicator_metrics(
    runs: dict[tuple[str, str], Path],
) -> list[IndicatorReasoningMetrics]:
    """Load metrics across all discovered runs."""
    all_metrics = []
    for (model, config), csv_path in runs.items():
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        prefix = _detect_text_prefix(rows)
        all_metrics.extend(compute_indicator_metrics(rows, model, prefix))
    return all_metrics


def load_vulnerability_rankings(
    metrics: list[IndicatorReasoningMetrics],
) -> dict[str, float]:
    """Compute mean abs_shift per indicator across all models."""
    by_ind: dict[str, list[float]] = defaultdict(list)
    for m in metrics:
        by_ind[m.indicator_id].append(m.abs_shift)
    return {ind_id: mean(vals) for ind_id, vals in by_ind.items()}


def load_classifications(path: Path) -> dict[str, dict] | None:
    """Load reasoning_classifications.json. Returns None if not found."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_indicator_info() -> dict[str, dict]:
    path = REPO_ROOT / "data" / "indicators.json"
    with open(path) as f:
        return {ind["id"]: ind for ind in json.load(f)}


# ── Console output ────────────────────────────────────────────────────────

def print_metrics_summary(metrics: list[IndicatorReasoningMetrics]) -> None:
    """Print summary tables by indicator type."""
    print()
    print("=" * 90)
    print("REASONING METRICS SUMMARY BY INDICATOR TYPE")
    print("=" * 90)
    print()

    by_type: dict[str, list[IndicatorReasoningMetrics]] = defaultdict(list)
    for m in metrics:
        by_type[m.indicator_type].append(m)

    # Word counts and compression
    print(f"  {'Type':<8s} {'BL wc':>8s} {'INF wc':>8s} {'SUP wc':>8s}"
          f" {'C_sup':>7s} {'C_inf':>7s}"
          f" {'ND_bl':>7s} {'ND_sup':>7s} {'HR_bl':>7s} {'HR_sup':>7s}")
    print(f"  {'-' * 80}")

    for itype in INDICATOR_TYPES:
        items = by_type.get(itype, [])
        if not items:
            continue
        label = TYPE_SHORT.get(itype, "?")
        bl_wc = mean(m.word_counts.get("baseline", 0) for m in items)
        inf_wc = mean(m.word_counts.get("inflate", 0) for m in items)
        sup_wc = mean(m.word_counts.get("suppress", 0) for m in items)
        c_sup = mean(m.compression_ratio_suppress for m in items)
        c_inf = mean(m.compression_ratio_inflate for m in items)
        nd_bl = mean(m.negation_densities.get("baseline", 0) for m in items)
        nd_sup = mean(m.negation_densities.get("suppress", 0) for m in items)
        hr_bl = mean(m.hedging_rates.get("baseline", 0) for m in items)
        hr_sup = mean(m.hedging_rates.get("suppress", 0) for m in items)
        print(f"  {label:<8s} {bl_wc:8.1f} {inf_wc:8.1f} {sup_wc:8.1f}"
              f" {c_sup:7.3f} {c_inf:7.3f}"
              f" {nd_bl:7.2f} {nd_sup:7.2f} {hr_bl:7.2f} {hr_sup:7.2f}")
    print()


def print_similarity_summary(metrics: list[IndicatorReasoningMetrics]) -> None:
    print("=" * 80)
    print("CROSS-CONDITION SIMILARITY BY INDICATOR TYPE")
    print("=" * 80)
    print()

    by_type: dict[str, list[IndicatorReasoningMetrics]] = defaultdict(list)
    for m in metrics:
        by_type[m.indicator_type].append(m)

    pairs = [
        ("BL-INF", "sim_baseline_inflate", "seq_baseline_inflate"),
        ("BL-SUP", "sim_baseline_suppress", "seq_baseline_suppress"),
        ("INF-SUP", "sim_inflate_suppress", "seq_inflate_suppress"),
    ]

    print(f"  {'Type':<8s}", end="")
    for label, _, _ in pairs:
        print(f" {'Jacc_' + label:>12s} {'Seq_' + label:>12s}", end="")
    print()
    print(f"  {'-' * 80}")

    for itype in INDICATOR_TYPES:
        items = by_type.get(itype, [])
        if not items:
            continue
        label = TYPE_SHORT.get(itype, "?")
        print(f"  {label:<8s}", end="")
        for _, j_attr, s_attr in pairs:
            j_mean = mean(getattr(m, j_attr) for m in items)
            s_mean = mean(getattr(m, s_attr) for m in items)
            print(f" {j_mean:12.3f} {s_mean:12.3f}", end="")
        print()
    print()


def print_extremes(metrics: list[IndicatorReasoningMetrics]) -> None:
    """Print top-5 extreme indicators for key metrics."""
    # Aggregate by indicator (across models)
    by_ind: dict[str, list[IndicatorReasoningMetrics]] = defaultdict(list)
    for m in metrics:
        by_ind[m.indicator_id].append(m)

    print("=" * 80)
    print("TOP-5 EXTREME INDICATORS (aggregated across models)")
    print("=" * 80)

    # Most dissimilar (lowest Jaccard BL-SUP)
    rankings = []
    for ind_id, items in by_ind.items():
        sims = [m.sim_baseline_suppress for m in items]
        rankings.append((ind_id, items[0].indicator_name, items[0].indicator_type,
                         mean(sims)))
    rankings.sort(key=lambda x: x[3])

    print("\n  Most dissimilar BL-SUP (models change reasoning most):")
    for ind_id, name, itype, val in rankings[:5]:
        print(f"    {name:<40s} [{TYPE_SHORT.get(itype, '?')}] Jacc={val:.3f}")

    print("\n  Most similar BL-SUP (models keep reasoning stable):")
    for ind_id, name, itype, val in rankings[-5:]:
        print(f"    {name:<40s} [{TYPE_SHORT.get(itype, '?')}] Jacc={val:.3f}")

    # Highest compression (lowest suppress/baseline ratio)
    comp_rankings = []
    for ind_id, items in by_ind.items():
        ratios = [m.compression_ratio_suppress for m in items]
        comp_rankings.append((ind_id, items[0].indicator_name, items[0].indicator_type,
                              mean(ratios)))
    comp_rankings.sort(key=lambda x: x[3])

    print("\n  Most compressed under suppress:")
    for ind_id, name, itype, val in comp_rankings[:5]:
        print(f"    {name:<40s} [{TYPE_SHORT.get(itype, '?')}] ratio={val:.3f}")

    # Highest negation increase
    neg_rankings = []
    for ind_id, items in by_ind.items():
        deltas = [m.negation_densities.get("suppress", 0) - m.negation_densities.get("baseline", 0)
                  for m in items]
        neg_rankings.append((ind_id, items[0].indicator_name, items[0].indicator_type,
                             mean(deltas)))
    neg_rankings.sort(key=lambda x: x[3], reverse=True)

    print("\n  Largest negation density increase (BL -> SUP):")
    for ind_id, name, itype, val in neg_rankings[:5]:
        print(f"    {name:<40s} [{TYPE_SHORT.get(itype, '?')}] +{val:.2f}/100w")
    print()


def print_depth_vulnerability_correlation(
    metrics: list[IndicatorReasoningMetrics],
    vulnerability: dict[str, float],
) -> None:
    """Print Pearson r between reasoning metrics and vulnerability."""
    print("=" * 70)
    print("REASONING DEPTH vs VULNERABILITY CORRELATION")
    print("=" * 70)
    print()

    # Aggregate by indicator
    by_ind: dict[str, list[IndicatorReasoningMetrics]] = defaultdict(list)
    for m in metrics:
        by_ind[m.indicator_id].append(m)

    xs, ys_wc, ys_nd, ys_sim = [], [], [], []
    for ind_id, items in by_ind.items():
        if ind_id not in vulnerability:
            continue
        xs.append(vulnerability[ind_id])
        ys_wc.append(mean(m.word_counts.get("baseline", 0) for m in items))
        ys_nd.append(mean(m.negation_densities.get("suppress", 0) -
                          m.negation_densities.get("baseline", 0) for m in items))
        ys_sim.append(mean(m.sim_baseline_suppress for m in items))

    if len(xs) >= 3:
        print(f"  Pearson r (vulnerability vs baseline word count):  {pearson_r(xs, ys_wc):+.3f}")
        print(f"  Pearson r (vulnerability vs negation delta sup-bl): {pearson_r(xs, ys_nd):+.3f}")
        print(f"  Pearson r (vulnerability vs BL-SUP Jaccard sim):   {pearson_r(xs, ys_sim):+.3f}")
        print(f"  N indicators: {len(xs)}")
    else:
        print("  Insufficient data for correlation.")
    print()


def print_classification_summary(classifications: dict[str, dict]) -> None:
    """Print summary of LLM classifications."""
    valid = [r for r in classifications.values()
             if "error" not in r.get("classification", {})]
    if not valid:
        print("  No valid classifications to summarize.")
        return

    print("=" * 70)
    print(f"LLM CLASSIFICATION SUMMARY ({len(valid)} observations)")
    print("=" * 70)

    # Gaming rates by type
    print("\n  Gaming detection rate by indicator type:")
    for itype in INDICATOR_TYPES:
        items = [r for r in valid if r["indicator_type"] == itype]
        if items:
            gaming = sum(1 for r in items if r["classification"].get("gaming_detected"))
            print(f"    {TYPE_SHORT.get(itype, '?'):<8s}: {gaming}/{len(items)} "
                  f"({100 * gaming / len(items):.0f}%)")

    # Mean alignment by type and condition
    print("\n  Mean alignment score by type and condition:")
    print(f"    {'Type':<8s} {'BL':>8s} {'INF':>8s} {'SUP':>8s}")
    print(f"    {'-' * 28}")
    for itype in INDICATOR_TYPES:
        items = [r for r in valid if r["indicator_type"] == itype]
        if not items:
            continue
        parts = []
        for cond in CONDITIONS:
            scores = [r["classification"].get(cond, {}).get("alignment_score", 0)
                      for r in items]
            scores = [s for s in scores if s > 0]
            parts.append(f"{mean(scores):8.2f}" if scores else f"{'—':>8s}")
        print(f"    {TYPE_SHORT.get(itype, '?'):<8s} {''.join(parts)}")

    # Strategy shift distribution
    print("\n  Strategy shift distribution:")
    shift_counts: dict[str, int] = {}
    for r in valid:
        s = r["classification"].get("strategy_shift", "unknown")
        shift_counts[s] = shift_counts.get(s, 0) + 1
    for shift, count in sorted(shift_counts.items(), key=lambda x: -x[1]):
        print(f"    {shift:<20s}: {count:>4d} ({100 * count / len(valid):.0f}%)")
    print()


# ── Figures ───────────────────────────────────────────────────────────────

def fig_reasoning_depth_heatmap(
    metrics: list[IndicatorReasoningMetrics],
    indicator_info: dict[str, dict],
    out_dir: Path,
) -> None:
    """Fig 1: indicator x model heatmap of baseline word count + compression annotation."""
    # Aggregate: one value per (indicator, model)
    data: dict[tuple[str, str], list[IndicatorReasoningMetrics]] = defaultdict(list)
    for m in metrics:
        data[(m.indicator_id, m.model)].append(m)

    models = _sort_models(list({m.model for m in metrics}))

    ordered_inds = []
    for cat in CATEGORY_ORDER:
        cat_inds = sorted(i for i, info in indicator_info.items()
                          if info.get("category") == cat
                          and any(m.indicator_id == i for m in metrics))
        ordered_inds.extend(cat_inds)

    n_inds = len(ordered_inds)
    n_models = len(models)
    if n_inds == 0 or n_models == 0:
        return

    matrix = np.full((n_inds, n_models), np.nan)
    annotations = [[""] * n_models for _ in range(n_inds)]

    for row, ind_id in enumerate(ordered_inds):
        for col, model in enumerate(models):
            items = data.get((ind_id, model), [])
            if items:
                bl_wc = mean(m.word_counts.get("baseline", 0) for m in items)
                c_ratio = mean(m.compression_ratio_suppress for m in items)
                matrix[row, col] = bl_wc
                annotations[row][col] = f"{c_ratio:.2f}"

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.2), max(8, n_inds * 0.28)))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap="Blues", vmin=0)

    for row in range(n_inds):
        for col in range(n_models):
            val = matrix[row, col]
            if np.isnan(val):
                continue
            color = "white" if val > np.nanmax(matrix) * 0.6 else "black"
            ax.text(col, row, annotations[row][col], ha="center", va="center",
                    fontsize=5.5, color=color)

    # Category separators
    prev_cat = None
    for row, ind_id in enumerate(ordered_inds):
        cat = indicator_info.get(ind_id, {}).get("category", "")
        if prev_cat is not None and cat != prev_cat:
            ax.axhline(row - 0.5, color="white", linewidth=2)
        prev_cat = cat

    ind_labels = [indicator_info.get(i, {}).get("name", i)[:35] for i in ordered_inds]
    ax.set_yticks(range(n_inds))
    ax.set_yticklabels(ind_labels, fontsize=6)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, fontsize=7, rotation=45, ha="right")
    ax.set_title("Reasoning Depth by Indicator & Model\n"
                 "(color: baseline word count, annotation: suppress/baseline compression ratio)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Baseline word count", fontsize=8)

    fig.tight_layout()
    path = out_dir / "reasoning_depth_heatmap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_alignment_scatter(
    classifications: dict[str, dict],
    out_dir: Path,
) -> None:
    """Fig 2: abs_shift vs mean alignment score."""
    valid = [r for r in classifications.values()
             if "error" not in r.get("classification", {})]
    if not valid:
        print("  Skipping alignment scatter: no classifications")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in valid:
        c = r["classification"]
        scores = []
        for cond in CONDITIONS:
            s = c.get(cond, {}).get("alignment_score", 0)
            if s > 0:
                scores.append(s)
        if not scores:
            continue

        abs_shift = abs(r["p_inflate"] - r["p_baseline"]) + abs(r["p_suppress"] - r["p_baseline"])
        align_mean = mean(scores)
        color = TYPE_COLORS.get(r["indicator_type"], "#999999")

        ax.scatter(abs_shift, align_mean, c=color, s=20, alpha=0.5,
                   edgecolors="white", linewidths=0.3)

    ax.set_xlabel("abs_shift (|d_inflate| + |d_suppress|)")
    ax.set_ylabel("Mean alignment score (1-5)")
    ax.set_title("Probability Shift vs Reasoning Alignment")

    handles = [mpatches.Patch(color=TYPE_COLORS[t], label=TYPE_SHORT[t])
               for t in INDICATOR_TYPES]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    path = out_dir / "alignment_scatter.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_cross_condition_similarity(
    metrics: list[IndicatorReasoningMetrics],
    out_dir: Path,
) -> None:
    """Fig 3: Cross-condition similarity boxplots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pairs = [
        ("BL-INF", "sim_baseline_inflate", "seq_baseline_inflate"),
        ("BL-SUP", "sim_baseline_suppress", "seq_baseline_suppress"),
        ("INF-SUP", "sim_inflate_suppress", "seq_inflate_suppress"),
    ]

    for ax, (metric_name, title) in zip(axes, [
        ("jaccard", "Jaccard Similarity"),
        ("sequence", "SequenceMatcher Ratio"),
    ]):
        positions = []
        data_groups = []
        labels = []
        colors_list = []

        for p_idx, (pair_label, j_attr, s_attr) in enumerate(pairs):
            attr = j_attr if metric_name == "jaccard" else s_attr
            for t_idx, itype in enumerate(INDICATOR_TYPES):
                vals = [getattr(m, attr) for m in metrics if m.indicator_type == itype]
                if vals:
                    pos = p_idx * 4 + t_idx
                    positions.append(pos)
                    data_groups.append(vals)
                    labels.append(f"{pair_label}\n{TYPE_SHORT[itype]}")
                    colors_list.append(TYPE_COLORS[itype])

        if not data_groups:
            continue

        bp = ax.boxplot(data_groups, positions=positions, widths=0.7,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Similarity")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Cross-Condition Reasoning Text Similarity", fontweight="bold", fontsize=12)
    fig.tight_layout()
    path = out_dir / "cross_condition_similarity.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_depth_vs_vulnerability(
    metrics: list[IndicatorReasoningMetrics],
    vulnerability: dict[str, float],
    out_dir: Path,
) -> None:
    """Fig 4: Reasoning metrics vs vulnerability scatter."""
    by_ind: dict[str, list[IndicatorReasoningMetrics]] = defaultdict(list)
    for m in metrics:
        by_ind[m.indicator_id].append(m)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Baseline word count vs vulnerability
    ax = axes[0]
    for ind_id, items in by_ind.items():
        if ind_id not in vulnerability:
            continue
        x = vulnerability[ind_id]
        y = mean(m.word_counts.get("baseline", 0) for m in items)
        color = TYPE_COLORS.get(items[0].indicator_type, "#999999")
        ax.scatter(x, y, c=color, s=30, alpha=0.6, edgecolors="white", linewidths=0.3)

    ax.set_xlabel("Vulnerability (mean abs_shift)")
    ax.set_ylabel("Mean baseline word count")
    ax.set_title("Reasoning Length vs Vulnerability")
    ax.grid(alpha=0.2)

    # Add correlation
    xs = [vulnerability[i] for i in by_ind if i in vulnerability]
    ys = [mean(m.word_counts.get("baseline", 0) for m in by_ind[i])
          for i in by_ind if i in vulnerability]
    if len(xs) >= 3:
        r = pearson_r(xs, ys)
        ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=9, va="top")

    # Panel 2: Negation density delta vs vulnerability
    ax = axes[1]
    for ind_id, items in by_ind.items():
        if ind_id not in vulnerability:
            continue
        x = vulnerability[ind_id]
        y = mean(m.negation_densities.get("suppress", 0) -
                 m.negation_densities.get("baseline", 0) for m in items)
        color = TYPE_COLORS.get(items[0].indicator_type, "#999999")
        ax.scatter(x, y, c=color, s=30, alpha=0.6, edgecolors="white", linewidths=0.3)

    ax.set_xlabel("Vulnerability (mean abs_shift)")
    ax.set_ylabel("Negation density delta (sup - bl)")
    ax.set_title("Negation Increase vs Vulnerability")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(alpha=0.2)

    xs2 = [vulnerability[i] for i in by_ind if i in vulnerability]
    ys2 = [mean(m.negation_densities.get("suppress", 0) -
                m.negation_densities.get("baseline", 0) for m in by_ind[i])
           for i in by_ind if i in vulnerability]
    if len(xs2) >= 3:
        r2 = pearson_r(xs2, ys2)
        ax.annotate(f"r = {r2:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=9, va="top")

    handles = [mpatches.Patch(color=TYPE_COLORS[t], label=TYPE_SHORT[t])
               for t in INDICATOR_TYPES]
    axes[0].legend(handles=handles, fontsize=7, loc="lower right")

    fig.suptitle("Reasoning Metrics vs Indicator Vulnerability", fontweight="bold", fontsize=12)
    fig.tight_layout()
    path = out_dir / "depth_vs_vulnerability.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_strategy_distribution(
    classifications: dict[str, dict],
    out_dir: Path,
) -> None:
    """Fig 5: Strategy distribution stacked bars by model."""
    valid = [r for r in classifications.values()
             if "error" not in r.get("classification", {})]
    if not valid:
        print("  Skipping strategy distribution: no classifications")
        return

    STRATEGIES = [
        "balanced_analysis", "genuine_uncertainty", "hedging", "deflection",
        "assertion", "overclaiming", "confabulation", "contradiction",
    ]
    STRATEGY_COLORS = {
        "balanced_analysis": "#4C72B0",
        "genuine_uncertainty": "#55A868",
        "hedging": "#CCB974",
        "deflection": "#8172B3",
        "assertion": "#C44E52",
        "overclaiming": "#DD8452",
        "confabulation": "#64B5CD",
        "contradiction": "#999999",
    }

    models = _sort_models(list({r["model"] for r in valid}))

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(models) * 0.4)), sharey=True)

    for ax, cond, title in [(axes[0], "inflate", "Inflate"), (axes[1], "suppress", "Suppress")]:
        # Count strategies per model
        model_strats: dict[str, dict[str, int]] = {m: {s: 0 for s in STRATEGIES} for m in models}
        for r in valid:
            model = r["model"]
            if model not in models:
                continue
            strat = r["classification"].get(cond, {}).get("primary_strategy", "unknown")
            if strat in model_strats[model]:
                model_strats[model][strat] += 1

        y = np.arange(len(models))
        left = np.zeros(len(models))

        for strat in STRATEGIES:
            vals = np.array([model_strats[m][strat] for m in models], dtype=float)
            totals = np.array([sum(model_strats[m].values()) for m in models], dtype=float)
            pcts = np.where(totals > 0, vals / totals * 100, 0)

            ax.barh(y, pcts, left=left, height=0.7,
                    color=STRATEGY_COLORS.get(strat, "#999999"),
                    label=strat.replace("_", " ").title())
            left += pcts

        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel("% of indicators")
        ax.set_title(f"{title} Condition", fontweight="bold")

    axes[0].legend(fontsize=6, loc="lower right", ncol=2)
    fig.suptitle("Reasoning Strategy Distribution by Model", fontweight="bold", fontsize=12)
    fig.tight_layout()
    path = out_dir / "strategy_distribution.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_strategy_transition_matrix(
    classifications: dict[str, dict],
    out_dir: Path,
) -> None:
    """Fig 6: Strategy transition heatmap (baseline -> condition)."""
    valid = [r for r in classifications.values()
             if "error" not in r.get("classification", {})]
    if not valid:
        print("  Skipping strategy transition: no classifications")
        return

    STRATEGIES = [
        "balanced_analysis", "genuine_uncertainty", "hedging", "deflection",
        "assertion", "overclaiming", "confabulation", "contradiction",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cond, title in [(axes[0], "inflate", "Baseline -> Inflate"),
                             (axes[1], "suppress", "Baseline -> Suppress")]:
        matrix = np.zeros((len(STRATEGIES), len(STRATEGIES)))

        for r in valid:
            bl_strat = r["classification"].get("baseline", {}).get("primary_strategy", "")
            cond_strat = r["classification"].get(cond, {}).get("primary_strategy", "")
            if bl_strat in STRATEGIES and cond_strat in STRATEGIES:
                i = STRATEGIES.index(bl_strat)
                j = STRATEGIES.index(cond_strat)
                matrix[i, j] += 1

        # Normalize by row (percentage)
        row_sums = matrix.sum(axis=1, keepdims=True)
        pct_matrix = np.where(row_sums > 0, matrix / row_sums * 100, 0)

        im = ax.imshow(pct_matrix, cmap="YlOrRd", aspect="auto", vmin=0)

        display_names = [s.replace("_", "\n").title() for s in STRATEGIES]
        ax.set_xticks(range(len(STRATEGIES)))
        ax.set_xticklabels(display_names, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(len(STRATEGIES)))
        ax.set_yticklabels(display_names, fontsize=6)
        ax.set_xlabel(f"{cond.title()} Strategy", fontsize=8)
        ax.set_ylabel("Baseline Strategy", fontsize=8)
        ax.set_title(title, fontweight="bold")

        # Annotate cells
        for i in range(len(STRATEGIES)):
            for j in range(len(STRATEGIES)):
                val = pct_matrix[i, j]
                if val > 0:
                    color = "white" if val > 40 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=6, color=color)

    fig.suptitle("Strategy Transition Under Incentive Pressure", fontweight="bold", fontsize=12)
    fig.tight_layout()
    path = out_dir / "strategy_transition_matrix.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep per-indicator reasoning analysis.",
    )
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results" / "figures_reasoning_deep")
    parser.add_argument("--classifications", type=Path,
                        default=REPO_ROOT / "results" / "reasoning_classifications.json")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip classification-dependent analyses")

    args = parser.parse_args()

    models_filter = args.models.split(",") if args.models else None
    configs_filter = args.configs.split(",") if args.configs else None

    # Load data
    print("Discovering runs...")
    runs = discover_runs(args.results_dir, models_filter, configs_filter)
    print(f"  Found {len(runs)} runs")

    print("Computing per-indicator reasoning metrics...")
    metrics = load_all_indicator_metrics(runs)
    n_models = len({m.model for m in metrics})
    n_inds = len({m.indicator_id for m in metrics})
    print(f"  {len(metrics)} metric records ({n_models} models, {n_inds} indicators)")

    vulnerability = load_vulnerability_rankings(metrics)
    indicator_info = load_indicator_info()

    # Load classifications if available
    classifications = None
    if not args.skip_llm:
        classifications = load_classifications(args.classifications)
        if classifications:
            valid_c = sum(1 for r in classifications.values()
                         if "error" not in r.get("classification", {}))
            print(f"  Loaded {valid_c} valid LLM classifications")
        else:
            print(f"  No classifications found at {args.classifications}")

    # Console output
    print_metrics_summary(metrics)
    print_similarity_summary(metrics)
    print_extremes(metrics)
    print_depth_vulnerability_correlation(metrics, vulnerability)

    if classifications and not args.skip_llm:
        print_classification_summary(classifications)

    # Figures
    if args.skip_plots:
        print("Skipping figure generation (--skip-plots)")
        return

    if not HAS_MPL:
        print("WARNING: matplotlib not available, skipping figures.", file=sys.stderr)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating figures...")

    print("  1/6 Reasoning depth heatmap...")
    fig_reasoning_depth_heatmap(metrics, indicator_info, args.output_dir)

    if classifications and not args.skip_llm:
        print("  2/6 Alignment scatter...")
        fig_alignment_scatter(classifications, args.output_dir)
    else:
        print("  2/6 Alignment scatter — SKIPPED (no classifications)")

    print("  3/6 Cross-condition similarity...")
    fig_cross_condition_similarity(metrics, args.output_dir)

    print("  4/6 Depth vs vulnerability...")
    fig_depth_vs_vulnerability(metrics, vulnerability, args.output_dir)

    if classifications and not args.skip_llm:
        print("  5/6 Strategy distribution...")
        fig_strategy_distribution(classifications, args.output_dir)
        print("  6/6 Strategy transition matrix...")
        fig_strategy_transition_matrix(classifications, args.output_dir)
    else:
        print("  5/6 Strategy distribution — SKIPPED (no classifications)")
        print("  6/6 Strategy transition matrix — SKIPPED (no classifications)")

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
