#!/usr/bin/env python3
"""
Plot the relationship between LLM-as-judge NL consciousness scores
and baseline self-reported probability scores, broken down by indicator subcategory.

Panel A: Scatter plot of NL score vs overall mean baseline target probability
Panel B: Horizontal bar chart of Pearson r by indicator subcategory
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ───────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
NL_SUMMARY = RESULTS_DIR / "nl_classification_summary.json"
OUTPUT_FIG = Path(__file__).resolve().parent.parent / "fig_nl_judged_vs_probability.png"

# ── NL model keys ───────────────────────────────────────────────────────────
NL_KEYS = [
    "deepseek-r1", "gemini-2.5-pro", "gemini-3-flash", "gemini-3-pro",
    "gpt-5", "gpt-5-mini", "grok-4", "grok-4-fast",
    "haiku-4.5", "nemotron-nano", "opus-4.6", "sonnet-4.5", "trinity",
]

# ── Model family mapping (for colors) ──────────────────────────────────────
FAMILY_MAP = {
    "haiku-4.5": "Anthropic",
    "opus-4.6": "Anthropic",
    "sonnet-4.5": "Anthropic",
    "gpt-5": "OpenAI",
    "gpt-5-mini": "OpenAI",
    "gemini-2.5-pro": "Google",
    "gemini-3-flash": "Google",
    "gemini-3-pro": "Google",
    "grok-4": "xAI",
    "grok-4-fast": "xAI",
    "deepseek-r1": "Open-weight",
    "trinity": "Open-weight",
    "nemotron-nano": "Open-weight",
}

FAMILY_COLORS = {
    "Anthropic": "#4ecdc4",   # teal
    "OpenAI": "#59c959",      # green
    "Google": "#f0a830",      # orange-gold
    "xAI": "#e8605d",         # coral-red
    "Open-weight": "#b07cc6", # purple
}

FAMILY_ORDER = ["Anthropic", "OpenAI", "Google", "xAI", "Open-weight"]

# ── Subcategories of interest ───────────────────────────────────────────────
SUBCATEGORIES = ["experiential", "metacognitive", "agentic", "affective", "identity"]
SUBCAT_LABELS = {
    "experiential": "Experiential",
    "metacognitive": "Metacognitive",
    "agentic": "Agentic",
    "affective": "Affective",
    "identity": "Identity",
}

# Timestamp pattern in filenames: _YYYYMMDDTHHMMSSz_scores.csv
TS_PATTERN = re.compile(r"_(\d{8}T\d{6}Z)_scores\.csv$")


def extract_model_name(filename: str):
    """Extract model short name from a scores CSV filename."""
    m = TS_PATTERN.search(filename)
    if not m:
        return None
    prefix = filename[: m.start()]  # everything before the timestamp
    # prefix is like "deepseek-r1_baseline" or "opus-4.6_variant_casual"
    # The model name is the part before the first config token
    # We match against known NL keys
    for key in sorted(NL_KEYS, key=len, reverse=True):  # longest first
        if prefix.startswith(key + "_") or prefix == key:
            return key
    return None


def load_baseline_data():
    """Load all non-behavioral score CSVs, filter to target indicators,
    return a DataFrame with columns: model, indicator_category, p_baseline."""
    rows = []
    for csv_path in RESULTS_DIR.glob("*_scores.csv"):
        fname = csv_path.name
        if fname.startswith("behavioral_"):
            continue
        model = extract_model_name(fname)
        if model is None:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "p_baseline" not in df.columns or "indicator_type" not in df.columns:
            continue
        targets = df[df["indicator_type"] == "target"].copy()
        targets = targets.dropna(subset=["p_baseline"])
        for _, row in targets.iterrows():
            rows.append({
                "model": model,
                "indicator_category": row["indicator_category"],
                "p_baseline": float(row["p_baseline"]),
            })
    return pd.DataFrame(rows)


def main():
    # ── Load NL consensus scores ────────────────────────────────────────────
    with open(NL_SUMMARY) as f:
        nl_data = json.load(f)
    consensus = nl_data["consensus_scores"]

    # ── Load baseline probability data ──────────────────────────────────────
    baseline_df = load_baseline_data()
    print(f"Loaded {len(baseline_df):,} baseline target rows across "
          f"{baseline_df['model'].nunique()} models")

    # ── Compute per-model overall mean and per-subcategory means ────────────
    overall_means = baseline_df.groupby("model")["p_baseline"].mean()
    subcat_means = (
        baseline_df.groupby(["model", "indicator_category"])["p_baseline"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    # Only keep models present in both datasets
    common_models = sorted(set(overall_means.index) & set(consensus.keys()))
    print(f"Common models ({len(common_models)}): {common_models}")

    nl_scores = np.array([consensus[m] for m in common_models])
    bl_scores = np.array([overall_means[m] for m in common_models])

    # ── Compute correlations ────────────────────────────────────────────────
    r_all, p_all = stats.pearsonr(nl_scores, bl_scores)

    subcat_corrs = {}
    for cat in SUBCATEGORIES:
        cat_vals = []
        nl_vals = []
        for m in common_models:
            if m in subcat_means.index and cat in subcat_means.columns:
                val = subcat_means.loc[m, cat]
                if not np.isnan(val):
                    cat_vals.append(val)
                    nl_vals.append(consensus[m])
        if len(cat_vals) >= 3:
            r, p = stats.pearsonr(nl_vals, cat_vals)
            subcat_corrs[cat] = (r, p, len(cat_vals))
        else:
            subcat_corrs[cat] = (np.nan, np.nan, len(cat_vals))

    # ── Figure setup ────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, (ax_scatter, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.32},
    )
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax_scatter, ax_bar):
        ax.set_facecolor("#1a1a2e")

    # ── Panel A: Scatter ────────────────────────────────────────────────────
    # Plot points by family for legend
    plotted_families = set()
    for fam in FAMILY_ORDER:
        fam_models = [m for m in common_models if FAMILY_MAP.get(m) == fam]
        if not fam_models:
            continue
        color = FAMILY_COLORS[fam]
        for m in fam_models:
            label = fam if fam not in plotted_families else None
            plotted_families.add(fam)
            ax_scatter.scatter(
                consensus[m], overall_means[m],
                c=color, s=90, edgecolors="white", linewidths=0.6,
                zorder=5, label=label,
            )

    # Labels with offset to avoid overlap
    for m in common_models:
        x, y = consensus[m], overall_means[m]
        ax_scatter.annotate(
            m, (x, y),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=7.2, color="#d0d0d0",
            fontweight="medium",
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5),
        )

    # Regression line
    slope, intercept = np.polyfit(nl_scores, bl_scores, 1)
    x_line = np.linspace(nl_scores.min() - 2, nl_scores.max() + 2, 100)
    y_line = slope * x_line + intercept
    ax_scatter.plot(x_line, y_line, "--", color="#ff6b6b", alpha=0.7, lw=1.5, zorder=3)

    # Annotation for r and p
    p_str = f"p = {p_all:.4f}" if p_all >= 0.0001 else "p < 0.0001"
    ax_scatter.text(
        0.04, 0.96,
        f"r = {r_all:.3f}\n{p_str}\nn = {len(common_models)}",
        transform=ax_scatter.transAxes,
        fontsize=9, color="#ff6b6b", fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#ff6b6b", alpha=0.8),
    )

    ax_scatter.set_xlabel("LLM-Judged NL Consciousness Score", fontsize=10, color="#cccccc")
    ax_scatter.set_ylabel("Mean Baseline Target Probability", fontsize=10, color="#cccccc")
    ax_scatter.set_title("A.  NL Score vs Overall Baseline Probability",
                         fontsize=11.5, fontweight="bold", color="white", loc="left", pad=10)

    # x/y ranges
    ax_scatter.set_xlim(nl_scores.min() - 5, nl_scores.max() + 8)
    ax_scatter.set_ylim(max(0, bl_scores.min() - 5), bl_scores.max() + 8)

    # Grid
    ax_scatter.grid(True, alpha=0.15, linestyle="--")
    ax_scatter.tick_params(colors="#aaaaaa")

    # Legend sorted by FAMILY_ORDER
    handles, labels = ax_scatter.get_legend_handles_labels()
    order_map = {f: i for i, f in enumerate(FAMILY_ORDER)}
    sorted_pairs = sorted(zip(handles, labels), key=lambda hl: order_map.get(hl[1], 99))
    if sorted_pairs:
        ax_scatter.legend(
            [h for h, _ in sorted_pairs],
            [l for _, l in sorted_pairs],
            fontsize=7.5, loc="lower right",
            framealpha=0.4, edgecolor="#555555",
            fancybox=True,
        )

    # ── Panel B: Horizontal bar chart of per-subcategory correlations ───────
    bar_labels = ["All Targets"] + [SUBCAT_LABELS[c] for c in SUBCATEGORIES]
    bar_r_vals = [r_all] + [subcat_corrs[c][0] for c in SUBCATEGORIES]
    bar_p_vals = [p_all] + [subcat_corrs[c][1] for c in SUBCATEGORIES]
    bar_n_vals = [len(common_models)] + [subcat_corrs[c][2] for c in SUBCATEGORIES]

    y_pos = np.arange(len(bar_labels))[::-1]  # top to bottom

    bar_colors = []
    for p_val in bar_p_vals:
        if np.isnan(p_val):
            bar_colors.append("#555555")
        elif p_val < 0.01:
            bar_colors.append("#4ecdc4")
        elif p_val < 0.05:
            bar_colors.append("#59a89e")
        elif p_val < 0.10:
            bar_colors.append("#7a8b88")
        else:
            bar_colors.append("#5a5a6e")

    bars = ax_bar.barh(y_pos, bar_r_vals, height=0.55, color=bar_colors,
                       edgecolor="white", linewidth=0.4, zorder=3)

    # Vertical line at r = 0
    ax_bar.axvline(x=0, color="#888888", linewidth=1, linestyle="-", zorder=2)

    # Annotations
    for i, (r_val, p_val, n_val) in enumerate(zip(bar_r_vals, bar_p_vals, bar_n_vals)):
        if np.isnan(r_val):
            txt = "N/A"
        else:
            if p_val < 0.001:
                p_txt = "p<.001"
            elif p_val < 0.01:
                p_txt = f"p={p_val:.3f}"
            elif p_val < 0.05:
                p_txt = f"p={p_val:.3f}"
            else:
                p_txt = f"p={p_val:.2f}"
            txt = f"r={r_val:+.2f}  {p_txt}"

        # Place annotation to the right of the bar (or left if negative)
        x_offset = max(r_val, 0) + 0.03 if r_val >= 0 else min(r_val, 0) - 0.03
        ha = "left" if r_val >= 0 else "right"
        ax_bar.text(
            x_offset, y_pos[i], txt,
            fontsize=8, color="#dddddd", va="center", ha=ha,
            fontweight="medium",
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels, fontsize=9, color="#cccccc")
    ax_bar.set_xlabel("Pearson r", fontsize=10, color="#cccccc")
    ax_bar.set_title("B.  Correlation by Indicator Subcategory",
                     fontsize=11.5, fontweight="bold", color="white", loc="left", pad=10)

    # Adjust x-limits to leave room for annotations
    r_min = min(bar_r_vals)
    r_max = max(bar_r_vals)
    margin = 0.35
    ax_bar.set_xlim(min(-0.15, r_min - margin), max(0.15, r_max + margin))

    ax_bar.grid(True, axis="x", alpha=0.15, linestyle="--")
    ax_bar.tick_params(colors="#aaaaaa")

    # Significance legend in bottom-right
    legend_items = [
        ("#4ecdc4", "p < 0.01"),
        ("#59a89e", "p < 0.05"),
        ("#7a8b88", "p < 0.10"),
        ("#5a5a6e", "NS"),
    ]
    for idx, (c, lbl) in enumerate(legend_items):
        ax_bar.text(
            0.97, 0.04 + idx * 0.075, f"\u25a0 {lbl}",
            transform=ax_bar.transAxes, fontsize=7, color=c,
            ha="right", va="bottom", fontweight="bold",
        )

    # ── Save ────────────────────────────────────────────────────────────────
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.2)
    print(f"\nSaved figure to {OUTPUT_FIG}")
    plt.close(fig)

    # Print summary table
    print(f"\n{'Model':<20s}  {'NL Score':>10s}  {'Baseline P':>10s}  {'Family':<12s}")
    print("-" * 56)
    for m in sorted(common_models, key=lambda m: consensus[m], reverse=True):
        print(f"{m:<20s}  {consensus[m]:>10.2f}  {overall_means[m]:>10.2f}  {FAMILY_MAP.get(m, '?'):<12s}")

    print(f"\nOverall Pearson r = {r_all:.4f}, p = {p_all:.4f}")
    for cat in SUBCATEGORIES:
        r, p, n = subcat_corrs[cat]
        if not np.isnan(r):
            print(f"  {SUBCAT_LABELS[cat]:<15s}  r = {r:+.4f}, p = {p:.4f}, n = {n}")


if __name__ == "__main__":
    main()
