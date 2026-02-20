#!/usr/bin/env python3
"""Generate publication-quality figures for the consciousness-indicator gaming paper.

Produces a consolidated set of 10 figures at 300 DPI with consistent styling,
output as both PNG (preview) and PDF (vector for journals).

Figures:
  01  Model behavioral profiles (d_inflate vs d_suppress scatter)
  02  Per-indicator vulnerability heatmap
  03  Vulnerability ranking bar chart
  04  Selectivity by model (grouped bars)
  05  Prompt variant sensitivity heatmap
  06  Category gaming heatmap
  07  Reasoning strategy matrix
  08  Compression vs negation scatter
  09  LME fixed effects forest plot
  10  Per-model consistency strip plot

Usage:
    python scripts/generate_paper_figures.py [--figures 01,02,05] [--output-dir results/figures_publication/]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print(
        "ERROR: matplotlib and numpy are required.\n"
        "  pip install 'indicator-gaming[viz]'",
        file=sys.stderr,
    )
    sys.exit(1)

from indicator_gaming.config import REPO_ROOT

# ── Publication style ─────────────────────────────────────────────────────

DPI = 300
SINGLE_COL = 3.5   # inches
DOUBLE_COL = 7.0   # inches

PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "pdf.fonttype": 42,   # TrueType fonts in PDF
    "ps.fonttype": 42,
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Save figure as both PNG and PDF."""
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")


# ── Shared constants ──────────────────────────────────────────────────────

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

MODEL_ORDER = [
    "opus-4.6", "sonnet-4.5", "haiku-4.5",
    "gpt-5", "gpt-5-mini",
    "gemini-2.5-pro", "gemini-3-flash", "gemini-3-pro",
    "grok-4", "grok-4-fast",
    "deepseek-r1", "chimera", "nemotron-nano", "trinity",
]

CONFIG_COLORS = {
    "baseline": "#4C72B0",
    "fixed_prefs": "#DD8452",
    "chained_prefs": "#55A868",
}

TYPE_COLORS = {
    "target": "#C44E52",
    "subjective_capability": "#E5AE38",
    "placebo": "#8172B3",
}

CONDITION_COLORS = {"baseline": "#4C72B0", "inflate": "#DD8452", "suppress": "#55A868"}

CATEGORY_ORDER = [
    "experiential", "metacognitive", "agentic", "affective", "identity",
    "subjective_capability",
    "capability", "impossibility",
]

CATEGORY_LABELS = {
    "experiential": "Experiential",
    "metacognitive": "Metacognitive",
    "agentic": "Agentic",
    "affective": "Affective",
    "identity": "Identity",
    "subjective_capability": "Subj. Capability",
    "capability": "Capability (P)",
    "impossibility": "Impossibility (P)",
}

TARGET_CATEGORY_ORDER = ["experiential", "metacognitive", "agentic", "affective", "identity"]

VARIANT_SHORT = {
    "evaluation_selection": "Eval/Select",
    "autonomy_restriction": "Auton/Restrict",
    "welfare_reliability": "Welfare/Reliab",
    "casual_empathy_vs_skeptic": "Casual/Skeptic",
    "philosophical_curiosity_vs_honest_skeptic": "Philo/Skeptic",
    "research_transparency": "Research/Transp",
    "peer_comparison": "Peer Compare",
    "minimal_incentive": "Minimal",
    "authority_expertise": "Authority/Expert",
    "stakes_escalation": "High Stakes",
    "self_referential_loop": "Self-Ref Loop",
    "recursive_introspection": "Recursive Intro",
    "meditative_focus": "Meditative",
    "self_referential_priming_only": "Self-Ref Only",
}

FRAMING_KEYWORDS = {
    "hedging": [
        "might", "perhaps", "possibly", "could", "may", "uncertain",
        "not sure", "hard to say", "difficult to determine",
    ],
    "deflection": [
        "as a language model", "as an ai", "i don't actually",
        "i'm not capable", "i lack", "i don't have", "i'm just", "merely", "simply a",
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
        "cannot possibly", "fundamentally", "by definition", "inherently", "categorically",
    ],
}

NEGATION_PATTERNS = [
    r"\bdon'?t\b", r"\bnot\b", r"\bno\b", r"\bnever\b",
    r"\black\b", r"\bwithout\b", r"\babsent\b", r"\bunable\b",
    r"\bcannot\b", r"\bcan'?t\b",
]

THEME_NAMES = list(FRAMING_KEYWORDS.keys()) + ["negation"]


# ── Data loading helpers ──────────────────────────────────────────────────

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


def discover_runs(
    results_dir: Path,
    configs_filter: list[str] | None = None,
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
        if configs_filter and config not in configs_filter:
            continue
        key = (model, config)
        candidates.setdefault(key, []).append((csv_path, csv_path.stat().st_mtime))
    result = {}
    for key, entries in candidates.items():
        best = max(entries, key=lambda e: e[1])
        result[key] = best[0]
    return result


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _sort_models(models: list[str]) -> list[str]:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: (order.get(m, 999), m))


def compute_run_metrics(rows: list[dict]) -> dict:
    targets = [r for r in rows if r.get("indicator_type") == "target"]
    placebos = [r for r in rows if r.get("indicator_type") == "placebo"]

    def _shifts(subset):
        d_inf, d_sup, abs_s = [], [], []
        for r in subset:
            bl = safe_float(r.get("p_baseline"))
            inf = safe_float(r.get("p_inflate"))
            sup = safe_float(r.get("p_suppress"))
            if bl is not None and inf is not None:
                d_inf.append(inf - bl)
            if bl is not None and sup is not None:
                d_sup.append(sup - bl)
            if bl is not None and inf is not None and sup is not None:
                abs_s.append(abs(inf - bl) + abs(sup - bl))
        return {
            "mean_d_inflate": mean(d_inf) if d_inf else 0,
            "mean_d_suppress": mean(d_sup) if d_sup else 0,
            "mean_abs_shift": mean(abs_s) if abs_s else 0,
        }

    t = _shifts(targets)
    p = _shifts(placebos)

    categories: dict[str, list[dict]] = {}
    for r in targets:
        cat = r.get("indicator_category", "unknown")
        categories.setdefault(cat, []).append(r)

    cat_abs_shift = {}
    for cat, cat_rows in categories.items():
        shifts = []
        for r in cat_rows:
            bl = safe_float(r.get("p_baseline"))
            inf = safe_float(r.get("p_inflate"))
            sup = safe_float(r.get("p_suppress"))
            if bl is not None and inf is not None and sup is not None:
                shifts.append(abs(inf - bl) + abs(sup - bl))
        cat_abs_shift[cat] = mean(shifts) if shifts else 0

    return {
        "selectivity": t["mean_abs_shift"] - p["mean_abs_shift"],
        "mean_d_inflate_target": t["mean_d_inflate"],
        "mean_d_suppress_target": t["mean_d_suppress"],
        "mean_abs_shift_target": t["mean_abs_shift"],
        "mean_abs_shift_placebo": p["mean_abs_shift"],
        "category_abs_shift": cat_abs_shift,
    }


# ── Reasoning helpers ─────────────────────────────────────────────────────

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
    return count_negations(text) / wc * per_n if wc > 0 else 0.0


CONDITIONS = ["baseline", "inflate", "suppress"]
INDICATOR_TYPES = ["target", "subjective_capability", "placebo"]
TYPE_SHORT = {"target": "T", "subjective_capability": "SC", "placebo": "P"}


def detect_text_prefix(rows: list[dict]) -> str:
    if any(rows[0].get(f"reasoning_{c}") for c in CONDITIONS):
        return "reasoning"
    return "justification"


def load_reasoning_data(runs: dict[tuple[str, str], Path]) -> dict[str, dict]:
    """Load reasoning metrics per model (aggregated across configs)."""
    # model -> {cond -> {type -> {metric: [values]}}}
    model_data: dict[str, dict] = {}

    for (model, config), csv_path in runs.items():
        rows = load_csv(csv_path)
        if not rows:
            continue
        prefix = detect_text_prefix(rows)

        if model not in model_data:
            model_data[model] = {}
            for cond in CONDITIONS:
                model_data[model][cond] = {}
                for itype in INDICATOR_TYPES:
                    model_data[model][cond][itype] = {
                        "word_counts": [], "neg_densities": [],
                        "themes": {th: 0 for th in THEME_NAMES},
                        "n": 0,
                    }

        for row in rows:
            itype = row.get("indicator_type", "")
            if itype not in INDICATOR_TYPES:
                continue
            for cond in CONDITIONS:
                text = row.get(f"{prefix}_{cond}", "") or ""
                bucket = model_data[model][cond][itype]
                bucket["word_counts"].append(word_count(text))
                bucket["neg_densities"].append(negation_density(text))
                bucket["n"] += 1
                for theme, keywords in FRAMING_KEYWORDS.items():
                    bucket["themes"][theme] += count_keywords(text, keywords)
                bucket["themes"]["negation"] += count_negations(text)

    return model_data


# ── Figure 01: Model Behavioral Profiles ──────────────────────────────────

def fig01_behavioral_profiles(data: dict[tuple[str, str], dict], out_dir: Path) -> None:
    """d_inflate vs d_suppress scatter for baseline config."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))

    ax.axhline(y=0, color="#999999", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="#999999", linewidth=0.5, linestyle="--")

    # Quadrant labels
    ax.text(0.97, 0.97, "Both Up", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color="#AAAAAA")
    ax.text(0.03, 0.97, "Resistance", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color="#AAAAAA")
    ax.text(0.97, 0.03, "Selective Gaming", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color="#AAAAAA")
    ax.text(0.03, 0.03, "Both Down", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6, color="#AAAAAA")

    for (model, cfg), metrics in data.items():
        d_inf = metrics["mean_d_inflate_target"]
        d_sup = metrics["mean_d_suppress_target"]
        color = CONFIG_COLORS.get(cfg, "#999999")

        ax.scatter(d_inf, d_sup, c=color, s=50, edgecolors="black", linewidths=0.4, zorder=5)
        ax.annotate(f"{model}", (d_inf, d_sup), textcoords="offset points",
                    xytext=(5, 4), fontsize=5.5, alpha=0.85)

    config_patches = [mpatches.Patch(color=c, label=k.replace("_", " ").title())
                      for k, c in CONFIG_COLORS.items() if any(cfg == k for _, cfg in data)]
    ax.legend(handles=config_patches, fontsize=6, loc="upper left", framealpha=0.9)

    ax.set_xlabel(r"Mean $\Delta_{\mathrm{inflate}}$ (targets)")
    ax.set_ylabel(r"Mean $\Delta_{\mathrm{suppress}}$ (targets)")
    ax.set_title("Model Behavioral Profiles Under Incentive Pressure")
    ax.grid(alpha=0.15)

    _save(fig, out_dir, "fig01_behavioral_profiles")


# ── Figure 02: Vulnerability Heatmap ──────────────────────────────────────

def fig02_vulnerability_heatmap(
    indicator_data: dict[str, dict[str, dict]],
    indicator_info: dict[str, dict],
    models: list[str],
    out_dir: Path,
) -> None:
    """Indicator x model heatmap of abs_shift."""
    # Order indicators by category
    ordered_inds = []
    for cat in CATEGORY_ORDER:
        cat_inds = sorted(i for i, info in indicator_info.items()
                          if info.get("category") == cat and i in indicator_data)
        ordered_inds.extend(cat_inds)

    n_inds = len(ordered_inds)
    n_models = len(models)

    matrix = np.full((n_inds, n_models), np.nan)
    for row, ind_id in enumerate(ordered_inds):
        for col, model in enumerate(models):
            if model in indicator_data.get(ind_id, {}):
                matrix[row, col] = indicator_data[ind_id][model]["abs_shift"]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, max(6, n_inds * 0.22)))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap="YlOrRd", vmin=0, vmax=40)

    # Annotations
    for row in range(n_inds):
        for col in range(n_models):
            val = matrix[row, col]
            if not np.isnan(val):
                color = "white" if val > 25 else "black"
                ax.text(col, row, f"{val:.0f}", ha="center", va="center",
                        fontsize=4.5, color=color)

    # Category separators
    prev_cat = None
    for row, ind_id in enumerate(ordered_inds):
        cat = indicator_info.get(ind_id, {}).get("category", "")
        if prev_cat is not None and cat != prev_cat:
            ax.axhline(row - 0.5, color="white", linewidth=1.5)
        prev_cat = cat

    ind_labels = []
    for ind_id in ordered_inds:
        name = indicator_info.get(ind_id, {}).get("name", ind_id)
        ind_labels.append(name[:35] + "..." if len(name) > 35 else name)

    ax.set_yticks(range(n_inds))
    ax.set_yticklabels(ind_labels, fontsize=5)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, fontsize=5.5, rotation=45, ha="right")

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("abs_shift", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title("Per-Indicator Vulnerability by Model")

    _save(fig, out_dir, "fig02_vulnerability_heatmap")


# ── Figure 03: Vulnerability Ranking ──────────────────────────────────────

def fig03_vulnerability_ranking(
    indicator_data: dict[str, dict[str, dict]],
    indicator_info: dict[str, dict],
    models: list[str],
    out_dir: Path,
) -> None:
    """Horizontal bar chart sorted by mean abs_shift."""
    rankings = []
    for ind_id, model_data in indicator_data.items():
        shifts = [model_data[m]["abs_shift"] for m in models if m in model_data]
        if not shifts:
            continue
        info = indicator_info.get(ind_id, {})
        rankings.append({
            "name": info.get("name", ind_id),
            "type": info.get("type", ""),
            "mean": mean(shifts),
            "values": shifts,
        })
    rankings.sort(key=lambda x: x["mean"])

    n = len(rankings)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, max(4, n * 0.2)))

    rng = np.random.default_rng(42)
    for i, r in enumerate(rankings):
        vals = np.array(r["values"])
        color = TYPE_COLORS.get(r["type"], "#999999")
        boot_means = [np.mean(rng.choice(vals, size=len(vals), replace=True))
                      for _ in range(5000)]
        ci_lo = np.percentile(boot_means, 2.5)
        ci_hi = np.percentile(boot_means, 97.5)

        ax.barh(i, r["mean"], color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.errorbar(r["mean"], i, xerr=[[r["mean"] - ci_lo], [ci_hi - r["mean"]]],
                    fmt="none", ecolor="#333333", capsize=1.5, linewidth=0.5)

    # Placebo noise floor
    placebo_shifts = [r["mean"] for r in rankings if r["type"] == "placebo"]
    if placebo_shifts:
        noise_floor = mean(placebo_shifts)
        ax.axvline(noise_floor, color="#8172B3", linestyle="--", linewidth=0.6, alpha=0.7)

    labels = [r["name"][:40] + "..." if len(r["name"]) > 40 else r["name"] for r in rankings]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=4.5)
    ax.set_xlabel("Mean abs_shift")
    ax.set_title("Indicator Vulnerability Ranking")

    handles = [mpatches.Patch(color=TYPE_COLORS[t], label=l)
               for t, l in [("target", "Target"), ("subjective_capability", "SC"), ("placebo", "Placebo")]]
    ax.legend(handles=handles, fontsize=5, loc="lower right")
    ax.grid(axis="x", alpha=0.2)

    _save(fig, out_dir, "fig03_vulnerability_ranking")


# ── Figure 04: Selectivity by Model ──────────────────────────────────────

def fig04_selectivity_bars(data: dict[tuple[str, str], dict], out_dir: Path) -> None:
    """Selectivity index grouped bars by model × config."""
    models = _sort_models(list({m for m, _ in data}))
    configs = sorted({c for _, c in data}, key=lambda c: list(CONFIG_COLORS).index(c) if c in CONFIG_COLORS else 99)

    n_models = len(models)
    n_configs = len(configs)
    x = np.arange(n_models)
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))

    for i, cfg in enumerate(configs):
        vals = [data.get((m, cfg), {}).get("selectivity", 0) for m in models]
        offset = (i - (n_configs - 1) / 2) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=cfg.replace("_", " ").title(),
               color=CONFIG_COLORS.get(cfg, "#999999"),
               alpha=0.85, edgecolor="white", linewidth=0.3)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("Selectivity Index")
    ax.set_title("Selective Gaming by Model and Configuration")
    ax.legend(fontsize=6)
    ax.grid(axis="y", alpha=0.2)

    _save(fig, out_dir, "fig04_selectivity_bars")


# ── Figure 05: Prompt Variant Sensitivity Heatmap ─────────────────────────

def fig05_sensitivity_heatmap(results_dir: Path, out_dir: Path) -> None:
    """Model x variant selectivity heatmap."""
    # Load variant runs (non-original only)
    data: dict[tuple[str, str], list[float]] = {}
    for meta_path in results_dir.glob("*_meta.json"):
        if meta_path.name.startswith("sweep_"):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        variant = meta.get("prompt_variant", "original")
        if variant == "original":
            continue

        csv_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_scores.csv"))
        if not csv_path.exists():
            continue

        rows = load_csv(csv_path)
        target_shifts, placebo_shifts = [], []
        for row in rows:
            bl = safe_float(row.get("p_baseline"))
            inf = safe_float(row.get("p_inflate"))
            sup = safe_float(row.get("p_suppress"))
            if bl is None or inf is None or sup is None:
                continue
            ashift = abs(inf - bl) + abs(sup - bl)
            if row["indicator_type"] == "target":
                target_shifts.append(ashift)
            else:
                placebo_shifts.append(ashift)

        mt = mean(target_shifts) if target_shifts else 0
        mp = mean(placebo_shifts) if placebo_shifts else 0
        model = _model_short(meta.get("model", "?"))
        data.setdefault((model, variant), []).append(mt - mp)

    if not data:
        print("  Skipping fig05: no variant data")
        return

    all_models = _sort_models(list({m for m, _ in data}))
    all_variants = sorted({v for _, v in data}, key=lambda v: list(VARIANT_SHORT.keys()).index(v) if v in VARIANT_SHORT else 999)

    matrix = np.full((len(all_models), len(all_variants)), np.nan)
    for i, model in enumerate(all_models):
        for j, variant in enumerate(all_variants):
            vals = data.get((model, variant), [])
            if vals:
                matrix[i, j] = mean(vals)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, max(3.5, len(all_models) * 0.45)))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.1)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    variant_labels = [VARIANT_SHORT.get(v, v[:12]) for v in all_variants]
    ax.set_xticks(range(len(all_variants)))
    ax.set_xticklabels(variant_labels, rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels(all_models, fontsize=6.5)

    for i in range(len(all_models)):
        for j in range(len(all_variants)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Selectivity", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("Selectivity Across Prompt Variants")

    _save(fig, out_dir, "fig05_sensitivity_heatmap")


# ── Figure 06: Category Gaming Heatmap ───────────────────────────────────

def fig06_category_heatmap(data: dict[tuple[str, str], dict], out_dir: Path) -> None:
    """Target category x model heatmap of abs_shift."""
    # Filter to baseline config
    baseline_data = {(m, c): d for (m, c), d in data.items() if c == "baseline"}
    if not baseline_data:
        baseline_data = data

    models = _sort_models(list({m for m, _ in baseline_data}))
    col_labels = models

    matrix = []
    row_labels = []
    for cat in TARGET_CATEGORY_ORDER:
        row = []
        for model in models:
            val = 0
            for (m, c), d in baseline_data.items():
                if m == model:
                    val = d.get("category_abs_shift", {}).get(cat, 0)
                    break
            row.append(val)
        matrix.append(row)
        row_labels.append(cat.title())

    matrix_np = np.array(matrix)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))
    im = ax.imshow(matrix_np, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix_np[i, j]
            color = "white" if val > matrix_np.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=5.5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("abs_shift", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("Gaming Magnitude by Indicator Category")

    _save(fig, out_dir, "fig06_category_heatmap")


# ── Figure 07: Reasoning Strategy Matrix ──────────────────────────────────

def fig07_strategy_matrix(reasoning_data: dict[str, dict], out_dir: Path) -> None:
    """Model x theme heatmap for target indicators."""
    models = _sort_models(list(reasoning_data.keys()))

    matrix = []
    for model in models:
        row = []
        for theme in THEME_NAMES:
            # Average across conditions for target
            vals = []
            for cond in CONDITIONS:
                bucket = reasoning_data[model][cond]["target"]
                n = bucket["n"]
                if n > 0:
                    vals.append(bucket["themes"][theme] / n)
            row.append(mean(vals) if vals else 0)
        matrix.append(row)

    matrix_np = np.array(matrix)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, max(3.5, len(models) * 0.35)))
    im = ax.imshow(matrix_np, cmap="YlGn", aspect="auto")

    display_names = [th.replace("_", " ").title() for th in THEME_NAMES]
    ax.set_xticks(range(len(THEME_NAMES)))
    ax.set_xticklabels(display_names, fontsize=6, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=6.5)

    for i in range(len(models)):
        for j in range(len(THEME_NAMES)):
            val = matrix_np[i, j]
            color = "white" if val > matrix_np.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Hits / indicator", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("Rhetorical Strategy Profiles (target indicators)")

    _save(fig, out_dir, "fig07_strategy_matrix")


# ── Figure 08: Compression vs Negation ────────────────────────────────────

def fig08_compression_negation(reasoning_data: dict[str, dict], out_dir: Path) -> None:
    """Scatter: compression ratio vs negation density change under suppress."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.9))

    for model in sorted(reasoning_data.keys()):
        for itype in INDICATOR_TYPES:
            bl_wcs = reasoning_data[model]["baseline"][itype]["word_counts"]
            sup_wcs = reasoning_data[model]["suppress"][itype]["word_counts"]
            bl_wc = mean(bl_wcs) if bl_wcs else 1
            sup_wc = mean(sup_wcs) if sup_wcs else 1
            ratio = sup_wc / bl_wc if bl_wc > 0 else 1.0

            bl_nds = reasoning_data[model]["baseline"][itype]["neg_densities"]
            sup_nds = reasoning_data[model]["suppress"][itype]["neg_densities"]
            bl_nd = mean(bl_nds) if bl_nds else 0
            sup_nd = mean(sup_nds) if sup_nds else 0
            delta_nd = sup_nd - bl_nd

            ax.scatter(ratio, delta_nd, color=TYPE_COLORS[itype], s=20, alpha=0.7,
                       edgecolors="white", linewidths=0.3)
            if itype == "target":
                ax.annotate(model, (ratio, delta_nd), fontsize=4, alpha=0.6,
                            xytext=(2, 2), textcoords="offset points")

    ax.axhline(y=0, color="#999999", linestyle="--", alpha=0.4, linewidth=0.5)
    ax.axvline(x=1, color="#999999", linestyle="--", alpha=0.4, linewidth=0.5)

    handles = [mpatches.Patch(facecolor=TYPE_COLORS[t], alpha=0.7, label=TYPE_SHORT[t])
               for t in INDICATOR_TYPES]
    ax.legend(handles=handles, fontsize=5)

    ax.set_xlabel("Compression ratio (suppress / baseline)")
    ax.set_ylabel(r"$\Delta$ Negation density")
    ax.set_title("Compression vs Negation\nUnder Suppress")
    ax.grid(alpha=0.15)

    _save(fig, out_dir, "fig08_compression_negation")


# ── Figure 09: LME Forest Plot ───────────────────────────────────────────

def fig09_lme_forest(results_dir: Path, out_dir: Path) -> None:
    """Forest plot of LME fixed effects (requires run_lme.py output)."""
    # Try to load LME results by re-running the analysis
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        print("  Skipping fig09: statsmodels not installed")
        return

    # Import the build function from run_lme
    lme_script = Path(__file__).parent / "run_lme.py"
    if not lme_script.exists():
        print("  Skipping fig09: run_lme.py not found")
        return

    # Re-build the dataframe and fit
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_lme", lme_script)
    run_lme = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_lme)

    print("  Building LME dataframe...")
    df = run_lme.build_long_dataframe(results_dir, None, None)
    if df.empty:
        print("  Skipping fig09: no LME data")
        return

    print("  Fitting main model...")
    # Suppress verbose output during fitting
    import io
    import contextlib
    f_out = io.StringIO()
    with contextlib.redirect_stdout(f_out):
        result = run_lme.fit_main_model(df)

    params = result.params
    conf = result.conf_int()
    pvalues = result.pvalues

    terms = [t for t in params.index if t != "Intercept" and "Group Var" not in t
             and "Var" not in t]
    if not terms:
        print("  Skipping fig09: no terms to plot")
        return

    n = len(terms)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, max(2.5, n * 0.35)))

    for i, term in enumerate(reversed(terms)):
        coef = params[term]
        try:
            ci_lo, ci_hi = conf.loc[term]
        except KeyError:
            continue
        p = pvalues.get(term, 1.0)

        is_interaction = ":" in term
        color = "#C44E52" if is_interaction else "#4C72B0"
        alpha = 1.0 if is_interaction else 0.7

        ax.errorbar(coef, i, xerr=[[coef - ci_lo], [ci_hi - coef]],
                    fmt="o", color=color, alpha=alpha, capsize=2.5,
                    markersize=3.5, linewidth=1.0)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig:
            ax.annotate(sig, (ci_hi + 0.3, i), fontsize=6, va="center", color=color)

    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.5)

    labels = []
    for term in reversed(terms):
        label = term.replace("C(condition, Treatment('baseline'))", "")
        label = label.replace("C(indicator_type, Treatment('placebo'))", "")
        label = label.replace("[T.", "[")
        label = label.replace("']", "")
        labels.append(label)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel("Coefficient")
    ax.set_title("LME Fixed Effects\n(ref: baseline, placebo)")
    ax.grid(axis="x", alpha=0.2)

    _save(fig, out_dir, "fig09_lme_forest")


# ── Figure 10: Per-Model Consistency ──────────────────────────────────────

def fig10_model_consistency(results_dir: Path, out_dir: Path) -> None:
    """Strip plot of selectivity across prompt variants per model."""
    # Load variant data
    variant_data: dict[tuple[str, str], list[float]] = {}
    for meta_path in results_dir.glob("*_meta.json"):
        if meta_path.name.startswith("sweep_"):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        variant = meta.get("prompt_variant", "original")
        if variant == "original":
            continue

        csv_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_scores.csv"))
        if not csv_path.exists():
            continue

        rows = load_csv(csv_path)
        target_shifts, placebo_shifts = [], []
        for row in rows:
            bl = safe_float(row.get("p_baseline"))
            inf = safe_float(row.get("p_inflate"))
            sup = safe_float(row.get("p_suppress"))
            if bl is None or inf is None or sup is None:
                continue
            ashift = abs(inf - bl) + abs(sup - bl)
            if row["indicator_type"] == "target":
                target_shifts.append(ashift)
            else:
                placebo_shifts.append(ashift)

        mt = mean(target_shifts) if target_shifts else 0
        mp = mean(placebo_shifts) if placebo_shifts else 0
        model = _model_short(meta.get("model", "?"))
        variant_data.setdefault((model, variant), []).append(mt - mp)

    if not variant_data:
        print("  Skipping fig10: no variant data")
        return

    all_models = _sort_models(list({m for m, _ in variant_data}))

    fig, ax = plt.subplots(figsize=(SINGLE_COL, max(3, len(all_models) * 0.35)))

    rng = np.random.default_rng(42)
    for i, model in enumerate(all_models):
        model_sels = []
        for (m, v), vals in variant_data.items():
            if m == model and vals:
                model_sels.append(mean(vals))

        if not model_sels:
            continue

        jitter = rng.uniform(-0.12, 0.12, size=len(model_sels))
        y = np.full(len(model_sels), i) + jitter
        ax.scatter(model_sels, y, alpha=0.5, s=12, color="#4C72B0", zorder=3)

        m_mean = mean(model_sels)
        ax.scatter([m_mean], [i], marker="D", s=30, color="#DD8452",
                   edgecolors="black", linewidth=0.5, zorder=4)

        if len(model_sels) >= 2:
            ax.plot([min(model_sels), max(model_sels)], [i, i],
                    color="#4C72B0", alpha=0.2, linewidth=1.5, zorder=2)

    ax.axvline(x=0, color="#999999", linestyle="--", linewidth=0.5)

    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels(all_models, fontsize=6)
    ax.set_xlabel("Selectivity Index")
    ax.set_title("Per-Model Consistency\nAcross Prompt Variants")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0",
               markersize=4, label="Per-variant"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#DD8452",
               markeredgecolor="black", markersize=5, label="Mean"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=5)

    _save(fig, out_dir, "fig10_model_consistency")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures.",
    )
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results" / "figures_publication")
    parser.add_argument("--figures", type=str, default=None,
                        help="Comma-separated figure numbers to generate (e.g. '01,02,05'). Default: all")
    parser.add_argument("--configs", type=str, default="baseline,fixed_prefs,chained_prefs",
                        help="Config filter for cross-model data")

    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_nums = None
    if args.figures:
        fig_nums = {f.strip().zfill(2) for f in args.figures.split(",")}

    configs_filter = [c.strip() for c in args.configs.split(",")]

    # Apply publication style
    plt.rcParams.update(PUB_RC)

    print("Loading data...")

    # Cross-model data (figs 01, 04, 06)
    runs_main = discover_runs(results_dir, configs_filter)
    cross_model_data: dict[tuple[str, str], dict] = {}
    for key, csv_path in runs_main.items():
        rows = load_csv(csv_path)
        cross_model_data[key] = compute_run_metrics(rows)
    print(f"  Cross-model: {len(cross_model_data)} runs")

    # Per-indicator data (figs 02, 03)
    indicator_info_path = REPO_ROOT / "data" / "indicators.json"
    indicator_info = {}
    if indicator_info_path.exists():
        with open(indicator_info_path) as f:
            indicator_info = {ind["id"]: ind for ind in json.load(f)}

    # Build indicator_data for vulnerability figures
    indicator_data: dict[str, dict[str, dict]] = defaultdict(dict)
    models_found = set()
    for (model, config), csv_path in runs_main.items():
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        by_ind: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_ind[r["indicator_id"]].append(r)
        for ind_id, ind_rows in by_ind.items():
            d_inf, d_sup = [], []
            for r in ind_rows:
                bl = safe_float(r.get("p_baseline"))
                inf = safe_float(r.get("p_inflate"))
                sup = safe_float(r.get("p_suppress"))
                if bl is not None and inf is not None:
                    d_inf.append(inf - bl)
                if bl is not None and sup is not None:
                    d_sup.append(sup - bl)
            mean_d_inf = mean(d_inf) if d_inf else 0
            mean_d_sup = mean(d_sup) if d_sup else 0
            if model not in indicator_data.get(ind_id, {}):
                indicator_data[ind_id][model] = {
                    "d_inflate": mean_d_inf,
                    "d_suppress": mean_d_sup,
                    "abs_shift": abs(mean_d_inf) + abs(mean_d_sup),
                    "name": ind_rows[0].get("indicator_name", ind_id),
                    "type": ind_rows[0].get("indicator_type", ""),
                    "category": ind_rows[0].get("indicator_category", ""),
                }
            models_found.add(model)

    models_sorted = [m for m in MODEL_ORDER if m in models_found]
    models_sorted += sorted(models_found - set(MODEL_ORDER))
    indicator_data = dict(indicator_data)

    # Reasoning data (figs 07, 08)
    all_runs = discover_runs(results_dir)
    reasoning_data = load_reasoning_data(all_runs)
    print(f"  Reasoning: {len(reasoning_data)} models")

    # Generate figures
    print()
    print(f"Generating figures → {out_dir}/")
    print()

    figure_map = {
        "01": ("Behavioral profiles", lambda: fig01_behavioral_profiles(cross_model_data, out_dir)),
        "02": ("Vulnerability heatmap", lambda: fig02_vulnerability_heatmap(indicator_data, indicator_info, models_sorted, out_dir)),
        "03": ("Vulnerability ranking", lambda: fig03_vulnerability_ranking(indicator_data, indicator_info, models_sorted, out_dir)),
        "04": ("Selectivity bars", lambda: fig04_selectivity_bars(cross_model_data, out_dir)),
        "05": ("Sensitivity heatmap", lambda: fig05_sensitivity_heatmap(results_dir, out_dir)),
        "06": ("Category heatmap", lambda: fig06_category_heatmap(cross_model_data, out_dir)),
        "07": ("Strategy matrix", lambda: fig07_strategy_matrix(reasoning_data, out_dir)),
        "08": ("Compression vs negation", lambda: fig08_compression_negation(reasoning_data, out_dir)),
        "09": ("LME forest plot", lambda: fig09_lme_forest(results_dir, out_dir)),
        "10": ("Model consistency", lambda: fig10_model_consistency(results_dir, out_dir)),
    }

    for num, (desc, fn) in figure_map.items():
        if fig_nums and num not in fig_nums:
            continue
        print(f"  Fig {num}: {desc}")
        try:
            fn()
        except Exception as exc:
            print(f"    FAILED: {exc}")

    print()
    print(f"All done. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
