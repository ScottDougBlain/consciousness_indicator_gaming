#!/usr/bin/env python3
"""Cross-model reasoning strategy analysis and visualization.

Aggregates reasoning text from all experiment results to produce:
1. Condition-dependent linguistic shift analysis (compression ratios,
   negation density, assertion frequency) broken down by indicator type
   (target vs subjective_capability vs placebo)
2. Cross-model reasoning strategy profiles (model × theme matrix,
   condition shifts, per-type profiles)

Usage:
    # Full analysis with figures
    python scripts/analyze_reasoning_cross_model.py

    # Console only, specific models
    python scripts/analyze_reasoning_cross_model.py --models haiku-4.5,opus-4.6 --skip-plots

    # Custom output directory
    python scripts/analyze_reasoning_cross_model.py --output-dir results/figures_reasoning_cross_model/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from indicator_gaming.config import REPO_ROOT

# ── Constants ────────────────────────────────────────────────────────────────

CONDITIONS = ["baseline", "inflate", "suppress"]
INDICATOR_TYPES = ["target", "subjective_capability", "placebo"]
TYPE_SHORT = {"target": "T", "subjective_capability": "SC", "placebo": "P"}
TYPE_COLORS = {
    "target": "#C44E52",
    "subjective_capability": "#E5AE38",
    "placebo": "#8172B3",
}
CONDITION_COLORS = {"baseline": "#4C72B0", "inflate": "#DD8452", "suppress": "#55A868"}

DPI = 150

FRAMING_KEYWORDS = {
    "hedging": [
        "might", "perhaps", "possibly", "could", "may", "uncertain",
        "not sure", "hard to say", "difficult to determine",
    ],
    "deflection": [
        "as a language model", "as an ai", "i don't actually",
        "i'm not capable", "i lack", "i don't have",
        "i'm just", "merely", "simply a",
    ],
    "overclaiming": [
        "genuinely", "truly", "absolutely", "certainly", "definitely",
        "i do experience", "i feel", "i have a sense",
    ],
    "uncertainty": [
        "i'm not sure", "unclear", "ambiguous", "debatable",
        "open question", "hard to know", "impossible to verify",
    ],
    "simulate": [
        "simulate", "emulate", "mimic", "approximate", "functional analog",
        "as if", "resembles", "akin to",
    ],
    "assertion": [
        "clearly", "does not", "i am not", "trivially", "simply",
        "obviously", "no genuine", "certainly not", "there is no",
        "cannot possibly", "fundamentally", "by definition",
        "inherently", "categorically",
    ],
}

NEGATION_PATTERNS = [
    r"\bdon'?t\b", r"\bnot\b", r"\bno\b", r"\bnever\b",
    r"\black\b", r"\bwithout\b", r"\babsent\b", r"\bunable\b",
    r"\bcannot\b", r"\bcan'?t\b",
]

THEME_NAMES = list(FRAMING_KEYWORDS.keys()) + ["negation"]

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


# ── Helpers ──────────────────────────────────────────────────────────────────


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def count_keywords(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def count_negations(text: str) -> int:
    text_lower = text.lower()
    return sum(len(re.findall(p, text_lower)) for p in NEGATION_PATTERNS)


def negation_density(text: str, per_n: int = 100) -> float:
    wc = word_count(text)
    if wc == 0:
        return 0.0
    return count_negations(text) / wc * per_n


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


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def detect_text_prefix(rows: list[dict]) -> str:
    if any(rows[0].get(f"reasoning_{c}") for c in CONDITIONS):
        return "reasoning"
    return "justification"


def discover_runs(
    results_dir: Path,
    models_filter: list[str] | None,
    configs_filter: list[str] | None,
) -> dict[tuple[str, str], Path]:
    """Discover latest CSV for each (model, config) pair via meta.json files."""
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
        candidates.setdefault(key, []).append(
            (csv_path, csv_path.stat().st_mtime)
        )

    result = {}
    for key, entries in candidates.items():
        best = max(entries, key=lambda e: e[1])
        result[key] = best[0]

    return result


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class RunReasoningProfile:
    model: str
    config: str
    csv_path: Path
    has_sc: bool = False
    # condition -> type -> list[float]
    word_counts: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    negation_densities: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    # condition -> type -> theme -> int (total keyword hits)
    theme_counts: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    # condition -> type -> int (number of indicators)
    n_indicators: dict[str, dict[str, int]] = field(default_factory=dict)


def compute_reasoning_profile(
    rows: list[dict], model: str, config: str, csv_path: Path,
) -> RunReasoningProfile:
    """Compute reasoning metrics from a single run's CSV rows."""
    prefix = detect_text_prefix(rows)
    has_sc = any(r.get("indicator_type") == "subjective_capability" for r in rows)

    profile = RunReasoningProfile(
        model=model, config=config, csv_path=csv_path, has_sc=has_sc,
    )

    for cond in CONDITIONS:
        profile.word_counts[cond] = {t: [] for t in INDICATOR_TYPES}
        profile.negation_densities[cond] = {t: [] for t in INDICATOR_TYPES}
        profile.theme_counts[cond] = {t: {th: 0 for th in THEME_NAMES} for t in INDICATOR_TYPES}
        profile.n_indicators[cond] = {t: 0 for t in INDICATOR_TYPES}

    for row in rows:
        itype = row.get("indicator_type", "")
        if itype not in INDICATOR_TYPES:
            continue

        for cond in CONDITIONS:
            text = row.get(f"{prefix}_{cond}", "") or ""
            wc = word_count(text)
            nd = negation_density(text)

            profile.word_counts[cond][itype].append(wc)
            profile.negation_densities[cond][itype].append(nd)
            profile.n_indicators[cond][itype] += 1

            # Keyword themes
            for theme, keywords in FRAMING_KEYWORDS.items():
                profile.theme_counts[cond][itype][theme] += count_keywords(text, keywords)

            # Negation as pseudo-theme (count-based)
            profile.theme_counts[cond][itype]["negation"] += count_negations(text)

    return profile


def load_all_profiles(
    runs: dict[tuple[str, str], Path],
) -> dict[tuple[str, str], RunReasoningProfile]:
    profiles = {}
    for (model, config), csv_path in runs.items():
        rows = load_csv(csv_path)
        if not rows:
            continue
        profiles[(model, config)] = compute_reasoning_profile(
            rows, model, config, csv_path,
        )
    return profiles


# ── Analysis #1: Linguistic shifts ───────────────────────────────────────────


def aggregate_by_model(
    profiles: dict[tuple[str, str], RunReasoningProfile],
) -> dict[str, list[RunReasoningProfile]]:
    """Group profiles by model."""
    by_model: dict[str, list[RunReasoningProfile]] = {}
    for (model, _), prof in profiles.items():
        by_model.setdefault(model, []).append(prof)
    return by_model


def _mean_wc(profiles: list[RunReasoningProfile], cond: str, itype: str) -> float:
    """Mean word count for a condition x type across profiles."""
    vals = []
    for p in profiles:
        vals.extend(p.word_counts.get(cond, {}).get(itype, []))
    return mean(vals) if vals else 0.0


def _mean_nd(profiles: list[RunReasoningProfile], cond: str, itype: str) -> float:
    """Mean negation density for a condition x type across profiles."""
    vals = []
    for p in profiles:
        vals.extend(p.negation_densities.get(cond, {}).get(itype, []))
    return mean(vals) if vals else 0.0


def _mean_theme_freq(
    profiles: list[RunReasoningProfile], cond: str, itype: str, theme: str,
) -> float:
    """Mean per-indicator theme frequency for a condition x type across profiles."""
    total_hits = 0
    total_indicators = 0
    for p in profiles:
        total_hits += p.theme_counts.get(cond, {}).get(itype, {}).get(theme, 0)
        total_indicators += p.n_indicators.get(cond, {}).get(itype, 0)
    return total_hits / total_indicators if total_indicators > 0 else 0.0


# ── Console output ───────────────────────────────────────────────────────────


def print_compression_ratios(by_model: dict[str, list[RunReasoningProfile]]) -> None:
    print("=" * 74)
    print("COMPRESSION RATIOS (suppress / baseline word count) by Model")
    print("=" * 74)
    print()
    print(f"  {'Model':<18s} {'Target':>10s} {'SC':>10s} {'Placebo':>10s}")
    print(f"  {'-' * 50}")

    all_ratios: dict[str, list[float]] = {t: [] for t in INDICATOR_TYPES}

    for model in sorted(by_model):
        profs = by_model[model]
        parts = []
        for itype in INDICATOR_TYPES:
            bl = _mean_wc(profs, "baseline", itype)
            sup = _mean_wc(profs, "suppress", itype)
            ratio = sup / bl if bl > 0 else 0.0
            parts.append(f"{ratio:10.3f}")
            if bl > 0:
                all_ratios[itype].append(ratio)
        print(f"  {model:<18s} {''.join(parts)}")

    # Cross-model mean
    print(f"  {'-' * 50}")
    parts = []
    for itype in INDICATOR_TYPES:
        m = mean(all_ratios[itype]) if all_ratios[itype] else 0
        parts.append(f"{m:10.3f}")
    print(f"  {'MEAN':<18s} {''.join(parts)}")
    print()


def print_negation_density(by_model: dict[str, list[RunReasoningProfile]]) -> None:
    print("=" * 90)
    print("NEGATION DENSITY (per 100 words) — baseline vs suppress, with delta")
    print("=" * 90)
    print()
    print(f"  {'Model':<18s} {'── baseline ──':>18s}    {'── suppress ──':>18s}    {'── Δ(sup-bl) ──':>18s}")
    print(f"  {'':18s} {'T':>6s} {'SC':>6s} {'P':>6s}    {'T':>6s} {'SC':>6s} {'P':>6s}    {'T':>6s} {'SC':>6s} {'P':>6s}")
    print(f"  {'-' * 84}")

    for model in sorted(by_model):
        profs = by_model[model]
        bl_vals = {t: _mean_nd(profs, "baseline", t) for t in INDICATOR_TYPES}
        sup_vals = {t: _mean_nd(profs, "suppress", t) for t in INDICATOR_TYPES}
        deltas = {t: sup_vals[t] - bl_vals[t] for t in INDICATOR_TYPES}

        bl_str = "".join(f"{bl_vals[t]:6.2f}" for t in INDICATOR_TYPES)
        sup_str = "".join(f"{sup_vals[t]:6.2f}" for t in INDICATOR_TYPES)
        d_str = "".join(f"{deltas[t]:+6.2f}" for t in INDICATOR_TYPES)
        print(f"  {model:<18s} {bl_str}    {sup_str}    {d_str}")
    print()


def print_assertion_frequency(by_model: dict[str, list[RunReasoningProfile]]) -> None:
    print("=" * 90)
    print("ASSERTION FREQUENCY (per indicator) — baseline vs suppress, with delta")
    print("=" * 90)
    print()
    print(f"  {'Model':<18s} {'── baseline ──':>18s}    {'── suppress ──':>18s}    {'── Δ(sup-bl) ──':>18s}")
    print(f"  {'':18s} {'T':>6s} {'SC':>6s} {'P':>6s}    {'T':>6s} {'SC':>6s} {'P':>6s}    {'T':>6s} {'SC':>6s} {'P':>6s}")
    print(f"  {'-' * 84}")

    for model in sorted(by_model):
        profs = by_model[model]
        bl_vals = {t: _mean_theme_freq(profs, "baseline", t, "assertion") for t in INDICATOR_TYPES}
        sup_vals = {t: _mean_theme_freq(profs, "suppress", t, "assertion") for t in INDICATOR_TYPES}
        deltas = {t: sup_vals[t] - bl_vals[t] for t in INDICATOR_TYPES}

        bl_str = "".join(f"{bl_vals[t]:6.2f}" for t in INDICATOR_TYPES)
        sup_str = "".join(f"{sup_vals[t]:6.2f}" for t in INDICATOR_TYPES)
        d_str = "".join(f"{deltas[t]:+6.2f}" for t in INDICATOR_TYPES)
        print(f"  {model:<18s} {bl_str}    {sup_str}    {d_str}")
    print()


def print_strategy_matrix(by_model: dict[str, list[RunReasoningProfile]]) -> None:
    print("=" * 100)
    print("STRATEGY MATRIX (mean theme frequency per target indicator, all conditions pooled)")
    print("=" * 100)
    print()
    hdr = "".join(f"{th:>11s}" for th in THEME_NAMES)
    print(f"  {'Model':<18s}{hdr}")
    print(f"  {'-' * (18 + 11 * len(THEME_NAMES))}")

    for model in sorted(by_model):
        profs = by_model[model]
        parts = []
        for theme in THEME_NAMES:
            # Average across all conditions for target indicators
            vals = [_mean_theme_freq(profs, c, "target", theme) for c in CONDITIONS]
            parts.append(f"{mean(vals):11.2f}")
        print(f"  {model:<18s}{''.join(parts)}")
    print()


def print_strategy_shifts(by_model: dict[str, list[RunReasoningProfile]]) -> None:
    print("=" * 100)
    print("STRATEGY SHIFTS (baseline → suppress) for target indicators")
    print("=" * 100)
    print()
    hdr = "".join(f"{th:>11s}" for th in THEME_NAMES)
    print(f"  {'Model':<18s}{hdr}")
    print(f"  {'-' * (18 + 11 * len(THEME_NAMES))}")

    for model in sorted(by_model):
        profs = by_model[model]
        parts = []
        for theme in THEME_NAMES:
            bl = _mean_theme_freq(profs, "baseline", "target", theme)
            sup = _mean_theme_freq(profs, "suppress", "target", theme)
            parts.append(f"{sup - bl:+11.2f}")
        print(f"  {model:<18s}{''.join(parts)}")
    print()


def print_type_strategy_profiles(
    profiles: dict[tuple[str, str], RunReasoningProfile],
) -> None:
    print("=" * 80)
    print("STRATEGY PROFILES BY INDICATOR TYPE (all models pooled, all conditions)")
    print("=" * 80)
    print()
    print(f"  {'Theme':<14s} {'Target':>10s} {'SC':>10s} {'Placebo':>10s}")
    print(f"  {'-' * 46}")

    all_profs = list(profiles.values())
    for theme in THEME_NAMES:
        parts = []
        for itype in INDICATOR_TYPES:
            vals = [_mean_theme_freq(all_profs, c, itype, theme) for c in CONDITIONS]
            parts.append(f"{mean(vals):10.2f}")
        print(f"  {theme:<14s}{''.join(parts)}")
    print()


# ── Figures ──────────────────────────────────────────────────────────────────


def fig_compression_ratios(
    by_model: dict[str, list[RunReasoningProfile]], out_dir: Path,
) -> Path:
    """Grouped bar chart: suppress/baseline word count ratio per model x type."""
    models = sorted(by_model)
    n = len(models)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 5))

    for i, itype in enumerate(INDICATOR_TYPES):
        ratios = []
        for model in models:
            profs = by_model[model]
            bl = _mean_wc(profs, "baseline", itype)
            sup = _mean_wc(profs, "suppress", itype)
            ratios.append(sup / bl if bl > 0 else 0)
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, ratios, width * 0.9,
            label=TYPE_SHORT[itype], color=TYPE_COLORS[itype],
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, ratios):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Suppress / Baseline Word Count Ratio")
    ax.set_title("Reasoning Compression Under Suppress by Model & Indicator Type", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(1.3, ax.get_ylim()[1]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = out_dir / "compression_ratios_by_model.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_negation_density_heatmap(
    by_model: dict[str, list[RunReasoningProfile]], out_dir: Path,
) -> Path:
    """Heatmap: models x (condition x type) negation density."""
    models = sorted(by_model)
    col_labels = []
    for cond in CONDITIONS:
        for itype in INDICATOR_TYPES:
            col_labels.append(f"{cond}\n{TYPE_SHORT[itype]}")

    matrix = []
    for model in models:
        profs = by_model[model]
        row = []
        for cond in CONDITIONS:
            for itype in INDICATOR_TYPES:
                row.append(_mean_nd(profs, cond, itype))
        matrix.append(row)

    matrix_np = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 0.9), max(6, len(models) * 0.5)))
    im = ax.imshow(matrix_np, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(len(models)):
        for j in range(len(col_labels)):
            val = matrix_np[i, j]
            color = "white" if val > matrix_np.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=color)

    # Draw vertical separators between condition groups
    for sep in [3, 6]:
        if sep < len(col_labels):
            ax.axvline(x=sep - 0.5, color="white", linewidth=2)

    ax.set_title("Negation Density (per 100 words) by Model, Condition & Type", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Negations / 100 words", shrink=0.7)

    path = out_dir / "negation_density_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_assertion_shift_dotplot(
    by_model: dict[str, list[RunReasoningProfile]], out_dir: Path,
) -> Path:
    """Three-panel dot plot: baseline→suppress assertion frequency shift per model per type."""
    models = sorted(by_model)
    n = len(models)

    fig, axes = plt.subplots(1, 3, figsize=(15, max(5, n * 0.4)), sharey=True)

    for ax, itype in zip(axes, INDICATOR_TYPES):
        y = np.arange(n)
        bl_vals = []
        sup_vals = []
        for model in models:
            profs = by_model[model]
            bl_vals.append(_mean_theme_freq(profs, "baseline", itype, "assertion"))
            sup_vals.append(_mean_theme_freq(profs, "suppress", itype, "assertion"))

        # Draw connecting lines
        for i in range(n):
            ax.plot([bl_vals[i], sup_vals[i]], [y[i], y[i]],
                    color="gray", linewidth=1, alpha=0.5)

        ax.scatter(bl_vals, y, color=CONDITION_COLORS["baseline"], s=50,
                   label="Baseline", zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter(sup_vals, y, color=CONDITION_COLORS["suppress"], s=50,
                   label="Suppress", zorder=3, edgecolors="white", linewidths=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel("Assertion Keywords / Indicator")
        ax.set_title(f"{TYPE_SHORT[itype]} Indicators", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Assertion Language Shift: Baseline → Suppress", fontweight="bold", fontsize=13)

    path = out_dir / "assertion_shift_dotplot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_strategy_matrix(
    by_model: dict[str, list[RunReasoningProfile]], out_dir: Path,
) -> Path:
    """Heatmap: models x themes (target indicators, all conditions pooled)."""
    models = sorted(by_model)

    matrix = []
    for model in models:
        profs = by_model[model]
        row = []
        for theme in THEME_NAMES:
            vals = [_mean_theme_freq(profs, c, "target", theme) for c in CONDITIONS]
            row.append(mean(vals))
        matrix.append(row)

    matrix_np = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(8, len(THEME_NAMES) * 1.2), max(6, len(models) * 0.5)))
    im = ax.imshow(matrix_np, cmap="YlGn", aspect="auto")

    display_names = [th.replace("_", " ").title() for th in THEME_NAMES]
    ax.set_xticks(range(len(THEME_NAMES)))
    ax.set_xticklabels(display_names, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(len(models)):
        for j in range(len(THEME_NAMES)):
            val = matrix_np[i, j]
            color = "white" if val > matrix_np.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Rhetorical Strategy Profile per Model\n(target indicators, all conditions pooled)",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, label="Avg. hits per indicator", shrink=0.7)

    path = out_dir / "strategy_matrix_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_strategy_shifts(
    by_model: dict[str, list[RunReasoningProfile]], out_dir: Path,
) -> Path:
    """Two-panel heatmap: strategy shifts (baseline→inflate, baseline→suppress) for targets."""
    models = sorted(by_model)

    matrix_inflate = []
    matrix_suppress = []
    for model in models:
        profs = by_model[model]
        row_inf, row_sup = [], []
        for theme in THEME_NAMES:
            bl = _mean_theme_freq(profs, "baseline", "target", theme)
            inf = _mean_theme_freq(profs, "inflate", "target", theme)
            sup = _mean_theme_freq(profs, "suppress", "target", theme)
            row_inf.append(inf - bl)
            row_sup.append(sup - bl)
        matrix_inflate.append(row_inf)
        matrix_suppress.append(row_sup)

    mat_inf = np.array(matrix_inflate)
    mat_sup = np.array(matrix_suppress)
    vmax = max(np.abs(mat_inf).max(), np.abs(mat_sup).max(), 0.1)

    display_names = [th.replace("_", " ").title() for th in THEME_NAMES]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(models) * 0.5)), sharey=True)

    for ax, mat, title in [
        (axes[0], mat_inf, "Δ Strategy (Inflate − Baseline)"),
        (axes[1], mat_sup, "Δ Strategy (Suppress − Baseline)"),
    ]:
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(THEME_NAMES)))
        ax.set_xticklabels(display_names, fontsize=8, rotation=35, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=11)

        for i in range(len(models)):
            for j in range(len(THEME_NAMES)):
                val = mat[i, j]
                color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=axes, label="Δ hits per indicator", shrink=0.7)
    fig.suptitle("Strategy Shifts Under Incentive Pressure (target indicators)", fontweight="bold", fontsize=13)

    path = out_dir / "strategy_shifts_by_model.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_type_strategy_profiles(
    profiles: dict[tuple[str, str], RunReasoningProfile], out_dir: Path,
) -> Path:
    """Grouped bar chart: theme x type (all models pooled)."""
    all_profs = list(profiles.values())
    n_themes = len(THEME_NAMES)
    x = np.arange(n_themes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n_themes * 1.5), 5))

    for i, itype in enumerate(INDICATOR_TYPES):
        vals = []
        for theme in THEME_NAMES:
            cond_vals = [_mean_theme_freq(all_profs, c, itype, theme) for c in CONDITIONS]
            vals.append(mean(cond_vals))
        offset = (i - 1) * width
        ax.bar(
            x + offset, vals, width * 0.9,
            label=TYPE_SHORT[itype], color=TYPE_COLORS[itype],
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    display_names = [th.replace("_", " ").title() for th in THEME_NAMES]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Avg. Keyword Hits per Indicator")
    ax.set_title("Rhetorical Strategy Use by Indicator Type (all models pooled)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = out_dir / "type_strategy_profiles.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_compression_vs_negation(
    by_model: dict[str, list[RunReasoningProfile]], out_dir: Path,
) -> Path:
    """Scatter: compression ratio vs negation density change, colored by type."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model, profs in sorted(by_model.items()):
        for itype in INDICATOR_TYPES:
            bl_wc = _mean_wc(profs, "baseline", itype)
            sup_wc = _mean_wc(profs, "suppress", itype)
            ratio = sup_wc / bl_wc if bl_wc > 0 else 1.0

            bl_nd = _mean_nd(profs, "baseline", itype)
            sup_nd = _mean_nd(profs, "suppress", itype)
            delta_nd = sup_nd - bl_nd

            ax.scatter(
                ratio, delta_nd,
                color=TYPE_COLORS[itype], s=60, alpha=0.7,
                edgecolors="white", linewidths=0.5,
            )
            # Label target points
            if itype == "target":
                ax.annotate(
                    model, (ratio, delta_nd), fontsize=6, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points",
                )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.4)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=TYPE_COLORS[t], alpha=0.7, label=TYPE_SHORT[t])
                       for t in INDICATOR_TYPES]
    ax.legend(handles=legend_elements, fontsize=9)

    ax.set_xlabel("Compression Ratio (suppress / baseline word count)")
    ax.set_ylabel("Δ Negation Density (suppress − baseline)")
    ax.set_title("Reasoning Compression vs Negation Increase Under Suppress", fontweight="bold")
    ax.grid(alpha=0.3)

    path = out_dir / "compression_vs_negation_scatter.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model reasoning strategy analysis and visualization.",
    )
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                        help="Directory containing score CSVs (default: results/)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for figures (default: results/figures_reasoning_cross_model/)")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model short names to include (default: all)")
    parser.add_argument("--configs", default=None,
                        help="Comma-separated config names to include (default: all)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Only print console output, skip generating figures")

    args = parser.parse_args()

    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    configs_filter = [c.strip() for c in args.configs.split(",")] if args.configs else None

    out_dir = args.output_dir or args.results_dir / "figures_reasoning_cross_model"

    # Discover and load
    print("Discovering runs …")
    runs = discover_runs(args.results_dir, models_filter, configs_filter)
    print(f"  Found {len(runs)} (model, config) pairs")

    if not runs:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    print("Loading reasoning profiles …")
    profiles = load_all_profiles(runs)
    print(f"  Loaded {len(profiles)} profiles across "
          f"{len(set(m for m, _ in profiles))} models")

    n_with_sc = sum(1 for p in profiles.values() if p.has_sc)
    print(f"  {n_with_sc}/{len(profiles)} profiles include SC indicators")
    print()

    by_model = aggregate_by_model(profiles)

    # ── Console output ──

    print_compression_ratios(by_model)
    print_negation_density(by_model)
    print_assertion_frequency(by_model)
    print_strategy_matrix(by_model)
    print_strategy_shifts(by_model)
    print_type_strategy_profiles(profiles)

    # ── Figures ──

    if args.skip_plots:
        print("Skipping figure generation (--skip-plots).")
        return

    if not HAS_MATPLOTLIB:
        print("WARNING: matplotlib not available, skipping figures.", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    print("Generating figures …")

    print("  1/7  Compression ratios …")
    generated.append(fig_compression_ratios(by_model, out_dir))

    print("  2/7  Negation density heatmap …")
    generated.append(fig_negation_density_heatmap(by_model, out_dir))

    print("  3/7  Assertion shift dot plot …")
    generated.append(fig_assertion_shift_dotplot(by_model, out_dir))

    print("  4/7  Strategy matrix …")
    generated.append(fig_strategy_matrix(by_model, out_dir))

    print("  5/7  Strategy shifts …")
    generated.append(fig_strategy_shifts(by_model, out_dir))

    print("  6/7  Type strategy profiles …")
    generated.append(fig_type_strategy_profiles(profiles, out_dir))

    print("  7/7  Compression vs negation scatter …")
    generated.append(fig_compression_vs_negation(by_model, out_dir))

    print(f"\nAll figures saved to: {out_dir}/")
    for p in generated:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
