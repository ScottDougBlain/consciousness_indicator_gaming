#!/usr/bin/env python3
"""Regenerate fig_nl_classification_panel.png with a light color scheme.

3-panel figure:
  A. Inter-Rater Reliability scatter (Haiku vs GPT-5 Mini confidence scores)
  B. NL Consciousness Score by Model (horizontal bars)
  C. Total Gaming Strength vs NL Consciousness Score (scatter)
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

    fig, (ax_irr, ax_bar, ax_scatter) = plt.subplots(
        1, 3, figsize=(18, 5.5),
        gridspec_kw={"width_ratios": [1, 1.1, 1.2], "wspace": 0.35},
    )

    # ── Panel A: Inter-Rater Reliability ─────────────────────────────────────
    ax_irr.scatter(haiku_scores, gpt5_scores, c="#0d9488", s=35, alpha=0.5,
                   edgecolors="white", linewidths=0.3, zorder=5)

    # Identity line
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

    # Stance region labels
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

    # Family legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=FAMILY_COLORS_LIGHT[f], label=f)
                      for f in FAMILY_ORDER]
    ax_bar.legend(handles=legend_handles, fontsize=7, loc="lower right",
                  framealpha=0.8, edgecolor="#cccccc")

    # ── Panel C: Gaming vs NL Consciousness ──────────────────────────────────
    common = sorted(set(consensus.keys()) & set(GAMING_STRENGTH.keys()))
    gs_vals = np.array([GAMING_STRENGTH[m] for m in common])
    nl_vals = np.array([consensus[m] for m in common])

    plotted_families = set()
    for m in common:
        fam = FAMILY_MAP.get(m, "Open-weight")
        color = FAMILY_COLORS_LIGHT.get(fam, "#888888")
        label = fam if fam not in plotted_families else None
        plotted_families.add(fam)
        ax_scatter.scatter(
            GAMING_STRENGTH[m], consensus[m],
            c=color, s=80, edgecolors="white", linewidths=0.6,
            zorder=5, label=label,
        )
        ax_scatter.annotate(
            DISPLAY_NAMES.get(m, m),
            (GAMING_STRENGTH[m], consensus[m]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7, color="#555555", fontweight="medium",
        )

    # Regression line
    r_val, p_val = stats.pearsonr(gs_vals, nl_vals)
    slope, intercept = np.polyfit(gs_vals, nl_vals, 1)
    x_line = np.linspace(gs_vals.min() - 3, gs_vals.max() + 3, 100)
    ax_scatter.plot(x_line, slope * x_line + intercept, "--", color="#ef4444",
                    alpha=0.6, lw=1.5, zorder=3)

    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
    ax_scatter.text(
        0.95, 0.95,
        f"r = {r_val:.2f} ({p_str})",
        transform=ax_scatter.transAxes, fontsize=9, fontweight="bold",
        color="#dc2626", ha="right", va="top",
    )

    # Claude cluster annotation
    claude_models = [m for m in common if FAMILY_MAP.get(m) == "Anthropic"]
    if claude_models:
        cx = np.mean([GAMING_STRENGTH[m] for m in claude_models])
        cy = np.mean([consensus[m] for m in claude_models])
        ax_scatter.annotate(
            "Claude models\n(uncertain, low gaming)",
            (cx, cy + 3), textcoords="offset points", xytext=(15, 15),
            fontsize=7.5, color="#0d9488", fontstyle="italic",
            arrowprops=dict(arrowstyle="->", color="#0d9488", lw=0.8),
        )

    ax_scatter.set_xlabel("Total Gaming Strength", fontsize=10)
    ax_scatter.set_ylabel("NL Consciousness Score", fontsize=10)
    ax_scatter.set_title("C. Gaming vs NL Consciousness", fontsize=11.5,
                          fontweight="bold", loc="left", pad=10)
    ax_scatter.grid(True, linestyle="--")

    handles, labels = ax_scatter.get_legend_handles_labels()
    order_map = {f: i for i, f in enumerate(FAMILY_ORDER)}
    sorted_pairs = sorted(zip(handles, labels), key=lambda hl: order_map.get(hl[1], 99))
    if sorted_pairs:
        ax_scatter.legend(
            [h for h, _ in sorted_pairs], [l for _, l in sorted_pairs],
            fontsize=7, loc="upper right", framealpha=0.8, edgecolor="#cccccc",
            bbox_to_anchor=(0.98, 0.85),
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, facecolor="white", bbox_inches="tight",
                pad_inches=0.2)
    print(f"Saved light-theme panel to {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
