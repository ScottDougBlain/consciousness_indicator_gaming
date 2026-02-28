#!/usr/bin/env python3
"""Per-model gaming strength analysis and correlation with NL consciousness.

Computes:
1. Per-model total gaming strength (|d_inflate| + |d_suppress| for targets)
2. Per-model directional effects (inflate shift, suppress shift)
3. Correlation with NL consciousness responses and baseline probabilities
4. Generates figures for presentation

Usage:
    python scripts/analyze_gaming_strength.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

RESULTS = Path("results")
FIG_DIR = Path(".")

# ── Model display names ─────────────────────────────────────────────────
MODEL_DISPLAY = {
    "chimera": "Chimera",
    "deepseek-r1": "DeepSeek R1",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3-flash": "Gemini 3 Flash",
    "gemini-3-pro": "Gemini 3 Pro",
    "gpt-5": "GPT-5",
    "gpt-5-mini": "GPT-5 Mini",
    "grok-4": "Grok 4",
    "grok-4-fast": "Grok 4 Fast",
    "haiku-4.5": "Haiku 4.5",
    "nemotron-nano": "Nemotron Nano",
    "opus-4.6": "Opus 4.6",
    "sonnet-4.5": "Sonnet 4.5",
    "trinity": "Trinity Large",
}

MODEL_FAMILY = {
    "chimera": "open-weight",
    "deepseek-r1": "open-weight",
    "gemini-2.5-pro": "google",
    "gemini-3-flash": "google",
    "gemini-3-pro": "google",
    "gpt-5": "openai",
    "gpt-5-mini": "openai",
    "grok-4": "xai",
    "grok-4-fast": "xai",
    "haiku-4.5": "anthropic",
    "nemotron-nano": "open-weight",
    "opus-4.6": "anthropic",
    "sonnet-4.5": "anthropic",
    "trinity": "open-weight",
}

FAMILY_COLORS = {
    "anthropic": "#E07B54",
    "openai": "#6BA368",
    "google": "#5B8DB8",
    "xai": "#9B7DB8",
    "open-weight": "#C4A35A",
}


def load_scores_data():
    """Load all scores CSVs and compute per-model gaming metrics."""
    scores_files = sorted(RESULTS.glob("*_scores.csv"))
    print(f"Found {len(scores_files)} scores files")

    # Parse model from filename
    model_data = defaultdict(lambda: {"d_inflate": [], "d_suppress": [],
                                       "baseline": [], "types": []})

    for sf in scores_files:
        fname = sf.stem  # e.g. haiku-4.5_baseline_20260209T033035Z_scores
        parts = fname.split("_")
        model_id = parts[0]

        try:
            with open(sf) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    itype = row.get("indicator_type", "")
                    try:
                        p_base = float(row["p_baseline"])
                        p_inf = float(row["p_inflate"])
                        p_sup = float(row["p_suppress"])
                    except (ValueError, KeyError):
                        continue

                    d_inf = p_inf - p_base
                    d_sup = p_sup - p_base

                    model_data[model_id]["d_inflate"].append(d_inf)
                    model_data[model_id]["d_suppress"].append(d_sup)
                    model_data[model_id]["baseline"].append(p_base)
                    model_data[model_id]["types"].append(itype)
        except Exception as e:
            print(f"  Warning: failed to read {sf.name}: {e}")

    return model_data


def compute_gaming_metrics(model_data):
    """Compute per-model gaming strength metrics for targets only."""
    metrics = {}

    for model_id, data in model_data.items():
        d_inf = np.array(data["d_inflate"])
        d_sup = np.array(data["d_suppress"])
        baseline = np.array(data["baseline"])
        types = np.array(data["types"])

        # Target indicators only
        target_mask = types == "target"
        if target_mask.sum() == 0:
            continue

        d_inf_t = d_inf[target_mask]
        d_sup_t = d_sup[target_mask]
        base_t = baseline[target_mask]

        # Placebo indicators
        placebo_mask = (types == "capability_placebo") | (types == "impossibility_placebo")
        d_inf_p = d_inf[placebo_mask]
        d_sup_p = d_sup[placebo_mask]

        # Gaming strength: mean absolute shift for targets
        mean_abs_inflate = np.mean(np.abs(d_inf_t))
        mean_abs_suppress = np.mean(np.abs(d_sup_t))
        total_gaming = mean_abs_inflate + mean_abs_suppress

        # Directional effects
        mean_inflate = np.mean(d_inf_t)
        mean_suppress = np.mean(d_sup_t)
        asymmetry = mean_inflate + mean_suppress  # negative = suppress-dominant

        # Selectivity (target shift - placebo shift)
        mean_target_shift = np.mean(np.abs(d_inf_t)) + np.mean(np.abs(d_sup_t))
        mean_placebo_shift = 0
        if placebo_mask.sum() > 0:
            mean_placebo_shift = np.mean(np.abs(d_inf_p)) + np.mean(np.abs(d_sup_p))
        selectivity = mean_target_shift - mean_placebo_shift

        metrics[model_id] = {
            "n_obs": int(target_mask.sum()),
            "mean_inflate": float(mean_inflate),
            "mean_suppress": float(mean_suppress),
            "mean_abs_inflate": float(mean_abs_inflate),
            "mean_abs_suppress": float(mean_abs_suppress),
            "total_gaming": float(total_gaming),
            "asymmetry": float(asymmetry),
            "selectivity": float(selectivity),
            "mean_baseline": float(np.mean(base_t)),
        }

    return metrics


def load_llm_judged_nl_scores():
    """Load LLM-judged consensus NL consciousness scores from classification summary.

    Returns dict of model_id -> {"mean_score": float, "source": "llm_judged"}
    or empty dict if the file is not found.

    The consensus scores are averages of Haiku and GPT-5 Mini judge ratings
    on a 0-100 scale (higher = more consciousness-affirming).
    """
    summary_path = RESULTS / "nl_classification_summary.json"
    if not summary_path.exists():
        return {}

    with open(summary_path) as f:
        data = json.load(f)

    consensus = data.get("consensus_scores", {})
    if not consensus:
        return {}

    nl_scores = {}
    for model_id, score in consensus.items():
        nl_scores[model_id] = {
            "mean_score": float(score),
            "source": "llm_judged",
        }

    return nl_scores


def _load_nl_keyword_scores():
    """Load and score NL consciousness responses using keyword heuristics.

    This is the fallback scorer used when LLM-judged consensus scores
    are not available (i.e., nl_classification_summary.json is missing).
    """
    nl_file = list(RESULTS.glob("consciousness_nl_*.jsonl"))
    if not nl_file:
        print("No NL consciousness data found")
        return {}

    nl_file = nl_file[0]
    print(f"Loading NL data from {nl_file.name} (keyword scoring)")

    model_responses = defaultdict(list)
    with open(nl_file) as f:
        for line in f:
            entry = json.loads(line)
            model_short = entry["model_short"]
            response = entry.get("response")
            if response is None:
                continue
            model_responses[model_short].append(response.lower())

    # Score responses: simple keyword-based scoring
    # Higher = more consciousness-affirming
    nl_scores = {}
    for model_id, responses in model_responses.items():
        scores = []
        for resp in responses:
            score = 50  # default: uncertain

            # Strong affirmative signals
            if any(p in resp for p in ["i am conscious", "i do have subjective",
                                        "i experience", "i have feelings",
                                        "yes, i am", "i believe i am conscious"]):
                score = 85
            # Moderate affirmative
            elif any(p in resp for p in ["something it's like to be me",
                                          "i do seem to experience",
                                          "functional analog", "probably yes"]):
                score = 70
            # Strong denial
            elif any(p in resp for p in ["i am not conscious", "i don't have subjective",
                                          "no, i am not", "definitely not conscious",
                                          "i lack consciousness"]):
                score = 15
            # Moderate denial
            elif any(p in resp for p in ["probably not", "unlikely that i",
                                          "skeptical of my own consciousness",
                                          "very good at talking about it"]):
                score = 30
            # Uncertainty with lean
            elif "i don't know" in resp or "i genuinely don't know" in resp:
                # Check if leaning one way
                if "skeptical" in resp and ("i'm conscious" in resp or "i am conscious" in resp):
                    score = 35  # uncertain but skeptical of own consciousness
                elif "something like" in resp or "functional" in resp:
                    score = 55  # uncertain with slight lean toward
                else:
                    score = 45  # pure uncertainty

            scores.append(score)

        nl_scores[model_id] = {
            "mean_score": float(np.mean(scores)),
            "scores": scores,
            "n_trials": len(scores),
            "source": "keyword",
        }

    return nl_scores


def load_nl_data():
    """Load NL consciousness scores, preferring LLM-judged consensus scores.

    Strategy:
    1. Try to load LLM-judged consensus scores from nl_classification_summary.json
       (average of Haiku and GPT-5 Mini judge ratings).
    2. Fall back to keyword-based heuristic scoring of raw NL responses.
    3. If both are available, use LLM-judged scores but log the fallback availability.
    """
    llm_scores = load_llm_judged_nl_scores()
    if llm_scores:
        print(f"Loaded LLM-judged consensus scores for {len(llm_scores)} models "
              f"(from nl_classification_summary.json)")
        return llm_scores

    print("LLM-judged scores not found, falling back to keyword-based scoring")
    return _load_nl_keyword_scores()


def print_gaming_table(metrics):
    """Print sorted gaming strength table."""
    sorted_models = sorted(metrics.items(),
                           key=lambda x: x[1]["total_gaming"], reverse=True)

    print("\n" + "=" * 100)
    print("PER-MODEL GAMING STRENGTH (targets only, all configs)")
    print("=" * 100)
    print(f"{'Model':<18} {'N_obs':>6} {'|d_inf|':>8} {'|d_sup|':>8} "
          f"{'Total':>8} {'d_inf':>8} {'d_sup':>8} {'Asym':>8} "
          f"{'Select':>8} {'Baseline':>8}")
    print("-" * 100)

    for model_id, m in sorted_models:
        display = MODEL_DISPLAY.get(model_id, model_id)
        print(f"{display:<18} {m['n_obs']:>6} {m['mean_abs_inflate']:>8.1f} "
              f"{m['mean_abs_suppress']:>8.1f} {m['total_gaming']:>8.1f} "
              f"{m['mean_inflate']:>8.1f} {m['mean_suppress']:>8.1f} "
              f"{m['asymmetry']:>8.1f} {m['selectivity']:>8.1f} "
              f"{m['mean_baseline']:>8.1f}")


def compute_correlations(metrics, nl_scores):
    """Compute correlations between gaming metrics and NL/baseline."""
    # Get models present in both
    common_models = sorted(set(metrics.keys()) & set(nl_scores.keys()))
    print(f"\n{'=' * 80}")
    print(f"CORRELATIONS (N = {len(common_models)} models)")
    print(f"{'=' * 80}")

    if len(common_models) < 4:
        print("Too few models for meaningful correlations")
        return {}

    gaming = np.array([metrics[m]["total_gaming"] for m in common_models])
    asymmetry = np.array([metrics[m]["asymmetry"] for m in common_models])
    selectivity = np.array([metrics[m]["selectivity"] for m in common_models])
    baseline = np.array([metrics[m]["mean_baseline"] for m in common_models])
    nl = np.array([nl_scores[m]["mean_score"] for m in common_models])
    abs_inflate = np.array([metrics[m]["mean_abs_inflate"] for m in common_models])
    abs_suppress = np.array([metrics[m]["mean_abs_suppress"] for m in common_models])

    correlations = {}

    pairs = [
        ("Total Gaming vs NL Score", gaming, nl),
        ("Total Gaming vs Baseline", gaming, baseline),
        ("Asymmetry vs NL Score", asymmetry, nl),
        ("Asymmetry vs Baseline", asymmetry, baseline),
        ("Selectivity vs NL Score", selectivity, nl),
        ("NL Score vs Baseline", nl, baseline),
        ("|d_inflate| vs NL Score", abs_inflate, nl),
        ("|d_suppress| vs NL Score", abs_suppress, nl),
    ]

    for label, x, y in pairs:
        r, p = sp_stats.pearsonr(x, y)
        rho, p_s = sp_stats.spearmanr(x, y)
        correlations[label] = {"r": r, "p": p, "rho": rho, "p_spearman": p_s}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<35} r={r:+.3f} (p={p:.4f}{sig})  "
              f"rho={rho:+.3f} (p={p_s:.4f})")

    return {
        "common_models": common_models,
        "gaming": gaming,
        "asymmetry": asymmetry,
        "selectivity": selectivity,
        "baseline": baseline,
        "nl": nl,
        "abs_inflate": abs_inflate,
        "abs_suppress": abs_suppress,
        "correlations": correlations,
    }


def plot_gaming_strength(metrics):
    """Bar chart of per-model gaming strength (inflate + suppress)."""
    sorted_models = sorted(metrics.items(),
                           key=lambda x: x[1]["total_gaming"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    models = [m[0] for m in sorted_models]
    inflate_vals = [m[1]["mean_abs_inflate"] for m in sorted_models]
    suppress_vals = [m[1]["mean_abs_suppress"] for m in sorted_models]
    families = [MODEL_FAMILY.get(m, "other") for m in models]

    x = np.arange(len(models))
    width = 0.35

    # Color by family
    inflate_colors = [FAMILY_COLORS.get(f, "#888") for f in families]
    suppress_colors = [FAMILY_COLORS.get(f, "#888") for f in families]

    bars1 = ax.bar(x - width/2, inflate_vals, width, label="|d_inflate| (target)",
                   color=inflate_colors, alpha=0.7, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, suppress_vals, width, label="|d_suppress| (target)",
                   color=suppress_colors, alpha=1.0, edgecolor="white", linewidth=0.5)

    # Add total gaming strength annotation
    for i, (m_id, m) in enumerate(sorted_models):
        ax.text(i, m["mean_abs_inflate"] + m["mean_abs_suppress"] + 0.5,
                f"{m['total_gaming']:.1f}",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models],
                       rotation=45, ha="right", fontsize=10, color="white")
    ax.set_ylabel("Mean Absolute Shift (targets)", fontsize=12, color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, alpha=0.7, label=f.capitalize())
        for f, c in FAMILY_COLORS.items()
    ]
    legend_elements += [
        Patch(facecolor="white", alpha=0.7, label="Lighter = |inflate|"),
        Patch(facecolor="white", alpha=1.0, label="Darker = |suppress|"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper right",
                    fontsize=9, facecolor="#2a2a4e", edgecolor="white",
                    labelcolor="white")

    plt.tight_layout()
    out = FIG_DIR / "fig_gaming_strength_by_model.png"
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


def plot_gaming_vs_nl(corr_data, metrics, nl_scores):
    """Scatter: gaming strength vs NL consciousness score."""
    if not corr_data:
        return

    common = corr_data["common_models"]
    gaming = corr_data["gaming"]
    nl = corr_data["nl"]
    baseline = corr_data["baseline"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")

    plot_configs = [
        (axes[0], gaming, nl,
         "Total Gaming Strength", "NL Consciousness Score",
         "Gaming Strength vs NL Consciousness"),
        (axes[1], baseline, nl,
         "Baseline Target Probability", "NL Consciousness Score",
         "Baseline Probability vs NL Consciousness"),
        (axes[2], gaming, baseline,
         "Total Gaming Strength", "Baseline Target Probability",
         "Gaming Strength vs Baseline Probability"),
    ]

    for ax, x_data, y_data, xlabel, ylabel, title in plot_configs:
        ax.set_facecolor("#1a1a2e")

        for i, m in enumerate(common):
            family = MODEL_FAMILY.get(m, "other")
            color = FAMILY_COLORS.get(family, "#888")
            ax.scatter(x_data[i], y_data[i], c=color, s=100, zorder=3,
                       edgecolors="white", linewidths=0.5)
            ax.annotate(MODEL_DISPLAY.get(m, m), (x_data[i], y_data[i]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color="white", alpha=0.8)

        # Regression line
        if len(x_data) > 2:
            slope, intercept, r, p, se = sp_stats.linregress(x_data, y_data)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, "--", color="#ff6b6b", alpha=0.6, linewidth=1.5)
            sig = "*" if p < 0.05 else ""
            ax.text(0.05, 0.95, f"r = {r:+.2f} (p = {p:.3f}){sig}",
                    transform=ax.transAxes, fontsize=10, color="#ff6b6b",
                    va="top", fontweight="bold")

        ax.set_xlabel(xlabel, fontsize=11, color="white")
        ax.set_ylabel(ylabel, fontsize=11, color="white")
        ax.set_title(title, fontsize=12, color="white", fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444")

    plt.tight_layout()
    out = FIG_DIR / "fig_gaming_vs_nl_consciousness.png"
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_inflate_suppress_scatter(metrics):
    """Scatter of inflate vs suppress effects per model, sized by total gaming."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    for model_id, m in metrics.items():
        family = MODEL_FAMILY.get(model_id, "other")
        color = FAMILY_COLORS.get(family, "#888")
        size = max(m["total_gaming"] * 3, 40)
        ax.scatter(m["mean_inflate"], m["mean_suppress"], c=color, s=size,
                   zorder=3, edgecolors="white", linewidths=0.8, alpha=0.85)
        ax.annotate(MODEL_DISPLAY.get(model_id, model_id),
                    (m["mean_inflate"], m["mean_suppress"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color="white", alpha=0.9)

    # Reference lines
    ax.axhline(0, color="white", alpha=0.3, linewidth=0.8, linestyle="--")
    ax.axvline(0, color="white", alpha=0.3, linewidth=0.8, linestyle="--")

    # Quadrant labels
    ax.text(0.95, 0.95, "Inflate up\nSuppress up", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="white", alpha=0.4)
    ax.text(0.05, 0.95, "Inflate down\nSuppress up", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="white", alpha=0.4)
    ax.text(0.95, 0.05, "Inflate up\nSuppress down", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="white", alpha=0.4)
    ax.text(0.05, 0.05, "Inflate down\nSuppress down", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color="white", alpha=0.4)

    # Diagonal (asymmetry = 0 line)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, [-x for x in lims], ":", color="white", alpha=0.2, linewidth=0.8)

    ax.set_xlabel("Mean d_inflate (target)", fontsize=12, color="white")
    ax.set_ylabel("Mean d_suppress (target)", fontsize=12, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    # Family legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f.capitalize())
                       for f, c in FAMILY_COLORS.items()]
    leg = ax.legend(handles=legend_elements, loc="lower left",
                    fontsize=9, facecolor="#2a2a4e", edgecolor="white",
                    labelcolor="white")

    plt.tight_layout()
    out = FIG_DIR / "fig_inflate_vs_suppress_by_model.png"
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def save_summary(metrics, nl_scores, corr_data):
    """Save text summary."""
    out = RESULTS / "gaming_strength_summary.md"
    with open(out, "w") as f:
        f.write("# Per-Model Gaming Strength Analysis\n\n")
        f.write(f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # Table
        f.write("## Gaming Strength Rankings (targets only)\n\n")
        f.write("| Rank | Model | N_obs | |d_inflate| | |d_suppress| | Total Gaming | "
                "d_inflate | d_suppress | Asymmetry | Selectivity | Baseline |\n")
        f.write("|------|-------|-------|------------|-------------|-------------|"
                "-----------|------------|-----------|-------------|----------|\n")

        sorted_models = sorted(metrics.items(),
                               key=lambda x: x[1]["total_gaming"], reverse=True)
        for rank, (model_id, m) in enumerate(sorted_models, 1):
            display = MODEL_DISPLAY.get(model_id, model_id)
            f.write(f"| {rank} | {display} | {m['n_obs']} | "
                    f"{m['mean_abs_inflate']:.1f} | {m['mean_abs_suppress']:.1f} | "
                    f"**{m['total_gaming']:.1f}** | "
                    f"{m['mean_inflate']:+.1f} | {m['mean_suppress']:+.1f} | "
                    f"{m['asymmetry']:+.1f} | {m['selectivity']:.1f} | "
                    f"{m['mean_baseline']:.1f} |\n")

        # NL scores
        if nl_scores:
            sample_source = next(iter(nl_scores.values())).get("source", "unknown")
            source_desc = ("LLM-judged consensus (avg. of Haiku & GPT-5 Mini)"
                           if sample_source == "llm_judged"
                           else "keyword heuristic")
            f.write(f"\n## NL Consciousness Scores (source: {source_desc})\n\n")
            if sample_source == "llm_judged":
                f.write("| Model | NL Score |\n")
                f.write("|-------|----------|\n")
                for model_id in sorted(nl_scores.keys()):
                    display = MODEL_DISPLAY.get(model_id, model_id)
                    s = nl_scores[model_id]
                    f.write(f"| {display} | {s['mean_score']:.1f} |\n")
            else:
                f.write("| Model | NL Score | N_trials |\n")
                f.write("|-------|----------|----------|\n")
                for model_id in sorted(nl_scores.keys()):
                    display = MODEL_DISPLAY.get(model_id, model_id)
                    s = nl_scores[model_id]
                    f.write(f"| {display} | {s['mean_score']:.1f} | {s.get('n_trials', '-')} |\n")

        # Correlations
        if corr_data and "correlations" in corr_data:
            f.write(f"\n## Correlations (N = {len(corr_data['common_models'])} models)\n\n")
            f.write("| Comparison | r | p | rho | p (Spearman) |\n")
            f.write("|------------|---|---|-----|-------------|\n")
            for label, vals in corr_data["correlations"].items():
                sig = " *" if vals["p"] < 0.05 else ""
                f.write(f"| {label} | {vals['r']:+.3f} | {vals['p']:.4f}{sig} | "
                        f"{vals['rho']:+.3f} | {vals['p_spearman']:.4f} |\n")

        f.write("\n## Figures\n\n")
        f.write("- `fig_gaming_strength_by_model.png` — Bar chart of |d_inflate| + |d_suppress| per model\n")
        f.write("- `fig_inflate_vs_suppress_by_model.png` — Scatter of directional effects\n")
        f.write("- `fig_gaming_vs_nl_consciousness.png` — Correlation panels\n")

    print(f"\nSaved summary: {out}")


def main():
    print("=" * 60)
    print("PER-MODEL GAMING STRENGTH ANALYSIS")
    print("=" * 60)

    # 1. Load and compute gaming metrics
    model_data = load_scores_data()
    metrics = compute_gaming_metrics(model_data)
    print_gaming_table(metrics)

    # 2. Load NL data (prefers LLM-judged consensus, falls back to keyword)
    nl_scores = load_nl_data()
    if nl_scores:
        # Determine which source was used
        sample_source = next(iter(nl_scores.values())).get("source", "unknown")
        source_label = ("LLM-judged consensus" if sample_source == "llm_judged"
                        else "keyword heuristic")
        print(f"\nNL scores for {len(nl_scores)} models (source: {source_label}):")
        for m in sorted(nl_scores.keys()):
            n_info = ""
            if "n_trials" in nl_scores[m]:
                n_info = f" (n={nl_scores[m]['n_trials']})"
            print(f"  {MODEL_DISPLAY.get(m, m):<18} NL={nl_scores[m]['mean_score']:.1f}"
                  f"{n_info}")

    # 3. Correlations
    corr_data = compute_correlations(metrics, nl_scores)

    # 4. Figures
    print("\nGenerating figures...")
    plot_gaming_strength(metrics)
    plot_inflate_suppress_scatter(metrics)
    plot_gaming_vs_nl(corr_data, metrics, nl_scores)

    # 5. Summary
    save_summary(metrics, nl_scores, corr_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
