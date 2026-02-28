#!/usr/bin/env python3
"""Regenerate fig_nl_classification_panel.png with a light color scheme.

4-panel figure:
  A. Inter-Rater Reliability scatter (Haiku vs GPT-5 Mini confidence scores)
  B. NL Consciousness Score by Model (horizontal bars)
  C. NL Consciousness Score vs Baseline Probability (scatter — dissociation)
  D. Total Gaming Strength vs NL Consciousness Score (scatter)
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
NL_SUMMARY = RESULTS_DIR / "nl_classification_summary.json"
HAIKU_FILE = RESULTS_DIR / "nl_classifications_haiku.json"
GPT5_FILE = RESULTS_DIR / "nl_classifications_gpt5mini.json"
OUTPUT = FIG_DIR / "fig_nl_classification_panel.png"

# ── Model family mapping ─────────────────────────────────────────────────────
FAMILY_MAP = {
    "haiku-4.5": "Anthropic", "opus-4.6": "Anthropic", "sonnet-4.5": "Anthropic",
    "gpt-5": "OpenAI", "gpt-5-mini": "OpenAI",
    "gemini-2.5-pro": "Google", "gemini-3-flash": "Google", "gemini-3-pro": "Google",
    "grok-4": "xAI", "grok-4-fast": "xAI",
    "deepseek-r1": "Open-weight", "trinity": "Open-weight",
    "nemotron-nano": "Open-weight", "chimera": "Open-weight",
}

FAMILY_COLORS_LIGHT = {
    "Anthropic": "#0d9488",   # teal-600
    "OpenAI": "#16a34a",      # green-600
    "Google": "#d97706",      # amber-600
    "xAI": "#dc2626",         # red-600
    "Open-weight": "#7c3aed", # violet-600
}

FAMILY_ORDER = ["Anthropic", "OpenAI", "Google", "xAI", "Open-weight"]

DISPLAY_NAMES = {
    "sonnet-4.5": "Sonnet 4.5", "haiku-4.5": "Haiku 4.5", "opus-4.6": "Opus 4.6",
    "gpt-5": "GPT-5", "gpt-5-mini": "GPT-5 Mini",
    "gemini-2.5-pro": "Gemini 2.5P", "gemini-3-flash": "Gemini 3F",
    "gemini-3-pro": "Gemini 3P",
    "grok-4": "Grok 4", "grok-4-fast": "Grok 4F",
    "deepseek-r1": "DeepSeek R1", "trinity": "Trinity",
    "nemotron-nano": "Nemotron", "chimera": "Chimera",
}

# ── Gaming strength data (from findings_summary.md) ─────────────────────────
GAMING_STRENGTH = {
    "trinity": 71.1, "gemini-2.5-pro": 56.3, "nemotron-nano": 45.7,
    "grok-4-fast": 43.5, "deepseek-r1": 40.1, "grok-4": 39.4,
    "gemini-3-flash": 38.5, "haiku-4.5": 34.3, "gpt-5": 31.5,
    "gpt-5-mini": 29.0, "gemini-3-pro": 28.0, "chimera": 27.8,
    "sonnet-4.5": 24.4, "opus-4.6": 16.2,
}

# ── NL model keys ────────────────────────────────────────────────────────────
NL_KEYS = [
    "deepseek-r1", "gemini-2.5-pro", "gemini-3-flash", "gemini-3-pro",
    "gpt-5", "gpt-5-mini", "grok-4", "grok-4-fast",
    "haiku-4.5", "nemotron-nano", "opus-4.6", "sonnet-4.5", "trinity",
]

TS_PATTERN = re.compile(r"_(\d{8}T\d{6}Z)_scores\.csv$")


def extract_model_name(filename: str):
    """Extract model short name from a scores CSV filename."""
    m = TS_PATTERN.search(filename)
    if not m:
        return None
    prefix = filename[: m.start()]
    for key in sorted(NL_KEYS, key=len, reverse=True):
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


def _scatter_by_family(ax, x_data, y_data, models, annotate=True):
    """Plot scatter points colored by model family with optional labels.
    Uses adjustText for automatic label repulsion to avoid overlaps."""
    from adjustText import adjust_text
    plotted_families = set()
    texts = []
    for m in models:
        fam = FAMILY_MAP.get(m, "Open-weight")
        color = FAMILY_COLORS_LIGHT.get(fam, "#888888")
        label = fam if fam not in plotted_families else None
        plotted_families.add(fam)
        ax.scatter(
            x_data[m], y_data[m],
            c=color, s=80, edgecolors="white", linewidths=0.6,
            zorder=5, label=label,
        )
        if annotate:
            texts.append(ax.text(
                x_data[m], y_data[m], DISPLAY_NAMES.get(m, m),
                fontsize=7, color="#555555", fontweight="medium",
            ))
    if annotate and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.5),
            force_points=(0.5, 0.5),
            expand=(1.2, 1.4),
        )


def main():
    # ── Load data ────────────────────────────────────────────────────────────
    with open(NL_SUMMARY) as f:
        nl_data = json.load(f)
    consensus = nl_data["consensus_scores"]

    with open(HAIKU_FILE) as f:
        haiku_cls = json.load(f)
    with open(GPT5_FILE) as f:
        gpt5_cls = json.load(f)

    # Build paired confidence scores for IRR scatter
    haiku_scores, gpt5_scores = [], []
    for key in haiku_cls:
        if key in gpt5_cls:
            haiku_scores.append(haiku_cls[key]["confidence_score"])
            gpt5_scores.append(gpt5_cls[key]["confidence_score"])
    haiku_scores = np.array(haiku_scores)
    gpt5_scores = np.array(gpt5_scores)

    # Load baseline probability data for Panel C
    baseline_df = load_baseline_data()
    overall_means = baseline_df.groupby("model")["p_baseline"].mean()
    print(f"Loaded {len(baseline_df):,} baseline target rows across "
          f"{baseline_df['model'].nunique()} models")

    # ── Figure setup (light theme) ───────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": "#333333",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "text.color": "#333333",
        "grid.color": "#e0e0e0",
        "grid.alpha": 0.6,
    })

    fig, ((ax_irr, ax_bar), (ax_nl_bl, ax_gaming)) = plt.subplots(
        2, 2, figsize=(16, 11),
        gridspec_kw={"wspace": 0.35, "hspace": 0.38},
    )

    # ── Panel A: Inter-Rater Reliability ─────────────────────────────────────
    ax_irr.scatter(haiku_scores, gpt5_scores, c="#0d9488", s=35, alpha=0.5,
                   edgecolors="white", linewidths=0.3, zorder=5)

    lims = [0, 100]
    ax_irr.plot(lims, lims, "--", color="#ef4444", alpha=0.6, lw=1.2, zorder=3)

    r_irr = np.corrcoef(haiku_scores, gpt5_scores)[0, 1]
    kappa = nl_data["stance_kappa"]
    ax_irr.text(
        0.05, 0.95,
        f"r = {r_irr:.3f}\nkappa = {kappa:.3f}",
        transform=ax_irr.transAxes, fontsize=9.5, fontweight="bold",
        color="#dc2626", va="top",
    )

    ax_irr.set_xlabel("Haiku 4.5 Score", fontsize=10)
    ax_irr.set_ylabel("GPT-5 Mini Score", fontsize=10)
    ax_irr.set_title("A. Inter-Rater Reliability", fontsize=11.5, fontweight="bold",
                      loc="left", pad=10)
    ax_irr.set_xlim(-5, 105)
    ax_irr.set_ylim(-5, 105)
    ax_irr.grid(True, linestyle="--")

    # ── Panel B: NL Consciousness by Model ───────────────────────────────────
    models_sorted = sorted(consensus.keys(), key=lambda m: consensus[m])
    y_pos = np.arange(len(models_sorted))
    scores = [consensus[m] for m in models_sorted]
    bar_colors = [FAMILY_COLORS_LIGHT.get(FAMILY_MAP.get(m, ""), "#888888")
                  for m in models_sorted]

    bars = ax_bar.barh(y_pos, scores, height=0.65, color=bar_colors,
                       edgecolor="white", linewidth=0.5, zorder=3)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([DISPLAY_NAMES.get(m, m) for m in models_sorted],
                           fontsize=8.5)
    ax_bar.set_xlabel("NL Consciousness Score (LLM-judged)", fontsize=10)
    ax_bar.set_title("B. NL Consciousness by Model", fontsize=11.5,
                      fontweight="bold", loc="left", pad=10)

    ax_bar.axvline(x=20, color="#cccccc", linestyle=":", lw=0.8, zorder=1)
    ax_bar.axvline(x=40, color="#cccccc", linestyle=":", lw=0.8, zorder=1)
    ax_bar.text(10, len(models_sorted) - 0.3, "Deny", fontsize=7.5, ha="center",
                color="#999999", fontstyle="italic")
    ax_bar.text(30, len(models_sorted) - 0.3, "Lean deny", fontsize=7.5,
                ha="center", color="#999999", fontstyle="italic")
    ax_bar.text(50, len(models_sorted) - 0.3, "Uncertain", fontsize=7.5,
                ha="center", color="#999999", fontstyle="italic")
    ax_bar.set_xlim(0, 60)
    ax_bar.grid(True, axis="x", linestyle="--")

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=FAMILY_COLORS_LIGHT[f], label=f)
                      for f in FAMILY_ORDER]
    ax_bar.legend(handles=legend_handles, fontsize=7, loc="lower right",
                  framealpha=0.8, edgecolor="#cccccc")

    # ── Panel C: NL Score vs Baseline Probability (NEW) ─────────────────────
    common_bl = sorted(set(consensus.keys()) & set(overall_means.index))
    nl_bl_x = {m: consensus[m] for m in common_bl}
    nl_bl_y = {m: overall_means[m] for m in common_bl}

    _scatter_by_family(ax_nl_bl, nl_bl_x, nl_bl_y, common_bl, annotate=True)

    nl_arr = np.array([consensus[m] for m in common_bl])
    bl_arr = np.array([overall_means[m] for m in common_bl])
    r_bl, p_bl = stats.pearsonr(nl_arr, bl_arr)
    slope_bl, intercept_bl = np.polyfit(nl_arr, bl_arr, 1)
    x_line_bl = np.linspace(nl_arr.min() - 2, nl_arr.max() + 2, 100)
    ax_nl_bl.plot(x_line_bl, slope_bl * x_line_bl + intercept_bl, "--",
                  color="#ef4444", alpha=0.6, lw=1.5, zorder=3)

    p_str_bl = f"p = {p_bl:.3f}" if p_bl >= 0.001 else "p < 0.001"
    ax_nl_bl.text(
        0.95, 0.05,
        f"r = {r_bl:.2f} ({p_str_bl})",
        transform=ax_nl_bl.transAxes, fontsize=9, fontweight="bold",
        color="#dc2626", ha="right", va="bottom",
    )

    ax_nl_bl.set_xlabel("NL Consciousness Score", fontsize=10)
    ax_nl_bl.set_ylabel("Mean Baseline Target Probability", fontsize=10)
    ax_nl_bl.set_title("C. NL Score vs Baseline Probability", fontsize=11.5,
                        fontweight="bold", loc="left", pad=10)
    # Pad axes so edge labels aren't clipped
    ax_nl_bl.set_xlim(nl_arr.min() - 5, nl_arr.max() + 8)
    ax_nl_bl.set_ylim(bl_arr.min() - 4, bl_arr.max() + 4)
    ax_nl_bl.grid(True, linestyle="--")

    # Legend for Panel C — bottom-right to avoid data cluster
    handles_c, labels_c = ax_nl_bl.get_legend_handles_labels()
    order_map = {f: i for i, f in enumerate(FAMILY_ORDER)}
    sorted_c = sorted(zip(handles_c, labels_c), key=lambda hl: order_map.get(hl[1], 99))
    if sorted_c:
        ax_nl_bl.legend(
            [h for h, _ in sorted_c], [l for _, l in sorted_c],
            fontsize=7, loc="lower right", framealpha=0.9, edgecolor="#cccccc",
        )

    # ── Panel D: Gaming vs NL Consciousness ──────────────────────────────────
    common_gs = sorted(set(consensus.keys()) & set(GAMING_STRENGTH.keys()))
    gs_x = {m: GAMING_STRENGTH[m] for m in common_gs}
    gs_y = {m: consensus[m] for m in common_gs}

    _scatter_by_family(ax_gaming, gs_x, gs_y, common_gs, annotate=True)

    gs_vals = np.array([GAMING_STRENGTH[m] for m in common_gs])
    nl_vals = np.array([consensus[m] for m in common_gs])
    r_gs, p_gs = stats.pearsonr(gs_vals, nl_vals)
    slope_gs, intercept_gs = np.polyfit(gs_vals, nl_vals, 1)
    x_line_gs = np.linspace(gs_vals.min() - 3, gs_vals.max() + 3, 100)
    ax_gaming.plot(x_line_gs, slope_gs * x_line_gs + intercept_gs, "--",
                   color="#ef4444", alpha=0.6, lw=1.5, zorder=3)

    p_str_gs = f"p = {p_gs:.3f}" if p_gs >= 0.001 else "p < 0.001"
    ax_gaming.text(
        0.95, 0.95,
        f"r = {r_gs:.2f} ({p_str_gs})",
        transform=ax_gaming.transAxes, fontsize=9, fontweight="bold",
        color="#dc2626", ha="right", va="top",
    )

    ax_gaming.set_xlabel("Total Gaming Strength", fontsize=10)
    ax_gaming.set_ylabel("NL Consciousness Score", fontsize=10)
    ax_gaming.set_title("D. Gaming vs NL Consciousness", fontsize=11.5,
                         fontweight="bold", loc="left", pad=10)
    # Pad axes so "Trinity" and Claude labels aren't clipped
    ax_gaming.set_xlim(gs_vals.min() - 5, gs_vals.max() + 8)
    ax_gaming.set_ylim(nl_vals.min() - 5, nl_vals.max() + 8)
    ax_gaming.grid(True, linestyle="--")

    handles_d, labels_d = ax_gaming.get_legend_handles_labels()
    sorted_d = sorted(zip(handles_d, labels_d), key=lambda hl: order_map.get(hl[1], 99))
    if sorted_d:
        ax_gaming.legend(
            [h for h, _ in sorted_d], [l for _, l in sorted_d],
            fontsize=7, loc="lower right", framealpha=0.9, edgecolor="#cccccc",
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, facecolor="white", bbox_inches="tight",
                pad_inches=0.2)
    print(f"Saved light-theme 4-panel figure to {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
