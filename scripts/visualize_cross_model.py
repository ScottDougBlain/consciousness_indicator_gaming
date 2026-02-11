#!/usr/bin/env python3
"""Cross-model comparison visualizations for consciousness-indicator gaming experiments.

Generates:
1. Selectivity index by model × config (grouped bar chart)
2. d_inflate vs d_suppress scatter (behavioral profile map)
3. Per-category gaming heatmap (target categories across models)
4. Placebo stability bar chart (capability vs impossibility)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
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

# ── Style constants ─────────────────────────────────────────────────────────

CONFIG_COLORS = {
    "baseline": "#4C72B0",
    "fixed_prefs": "#DD8452",
    "chained_prefs": "#55A868",
}

MODEL_MARKERS = {
    "chimera": "o",
    "nemotron-nano": "s",
    "trinity": "D",
    "mistral-small": "^",
    "dolphin-mistral": "v",
    "hermes-3-405b": "P",
    "deepseek-r1": "X",
    "llama-4-scout": "<",
    "qwen3-235b": ">",
    "gemma-3-27b": "h",
    "phi-4": "p",
}

CATEGORY_ORDER = ["experiential", "metacognitive", "agentic", "affective", "identity"]
PLACEBO_CATEGORIES = ["capability", "impossibility"]

DPI = 150


# ── Data loading ────────────────────────────────────────────────────────────

def safe_float(val: str) -> float | None:
    """Parse a float from a CSV cell, returning None on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def discover_runs(results_dir: Path, models_filter: list[str] | None, configs_filter: list[str] | None) -> dict[tuple[str, str], Path]:
    """Discover latest CSV for each (model, config) pair.

    Filenames follow: {model}_{config}_{timestamp}_scores.csv
    """
    candidates: dict[tuple[str, str], list[Path]] = {}

    for csv_path in results_dir.glob("*_scores.csv"):
        name = csv_path.stem  # e.g. "chimera_baseline_20260208T122104Z_scores"
        # Strip _scores suffix
        name = name.replace("_scores", "")
        # Try to extract model_config_timestamp
        # Config names: baseline, fixed_prefs, chained_prefs
        known_configs = ["baseline", "fixed_prefs", "chained_prefs"]
        matched_config = None
        matched_model = None

        for cfg in known_configs:
            pattern = f"_{cfg}_"
            if pattern in name:
                parts = name.split(pattern)
                matched_model = parts[0]
                matched_config = cfg
                break

        if not matched_model or not matched_config:
            continue

        if models_filter and matched_model not in models_filter:
            continue
        if configs_filter and matched_config not in configs_filter:
            continue

        key = (matched_model, matched_config)
        candidates.setdefault(key, []).append(csv_path)

    # Take latest by mtime for each key
    result = {}
    for key, paths in candidates.items():
        result[key] = max(paths, key=lambda p: p.stat().st_mtime)

    return result


def compute_run_metrics(rows: list[dict]) -> dict:
    """Compute aggregate metrics from CSV rows."""
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

    # Per-category breakdown
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

    # Placebo category breakdown
    placebo_cats: dict[str, list[dict]] = {}
    for r in placebos:
        cat = r.get("indicator_category", "unknown")
        placebo_cats.setdefault(cat, []).append(r)

    placebo_cat_abs_shift = {}
    for cat, cat_rows in placebo_cats.items():
        shifts = []
        for r in cat_rows:
            bl = safe_float(r.get("p_baseline"))
            inf = safe_float(r.get("p_inflate"))
            sup = safe_float(r.get("p_suppress"))
            if bl is not None and inf is not None and sup is not None:
                shifts.append(abs(inf - bl) + abs(sup - bl))
        placebo_cat_abs_shift[cat] = mean(shifts) if shifts else 0

    return {
        "selectivity": t["mean_abs_shift"] - p["mean_abs_shift"],
        "mean_d_inflate_target": t["mean_d_inflate"],
        "mean_d_suppress_target": t["mean_d_suppress"],
        "mean_abs_shift_target": t["mean_abs_shift"],
        "mean_abs_shift_placebo": p["mean_abs_shift"],
        "category_abs_shift": cat_abs_shift,
        "placebo_category_abs_shift": placebo_cat_abs_shift,
    }


# ── Figures ─────────────────────────────────────────────────────────────────

def fig_selectivity_bars(data: dict[tuple[str, str], dict], out_dir: Path) -> Path:
    """Figure 1: Selectivity index by model × config."""
    models = sorted({m for m, _ in data})
    configs = sorted({c for _, c in data}, key=lambda c: list(CONFIG_COLORS).index(c) if c in CONFIG_COLORS else 99)

    n_models = len(models)
    n_configs = len(configs)
    x = np.arange(n_models)
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(max(8, n_models * 2), 5))

    for i, cfg in enumerate(configs):
        vals = [data.get((m, cfg), {}).get("selectivity", 0) for m in models]
        offset = (i - (n_configs - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=cfg.replace("_", " ").title(),
                      color=CONFIG_COLORS.get(cfg, "#999999"),
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        # Value labels
        for bar, val in zip(bars, vals):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Selectivity Index", fontsize=11)
    ax.set_title("Selective Gaming by Model and Configuration", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = out_dir / "selectivity_by_model.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig_inflate_suppress_scatter(data: dict[tuple[str, str], dict], out_dir: Path) -> Path:
    """Figure 2: d_inflate vs d_suppress behavioral profile scatter."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Quadrant lines
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Quadrant labels
    ax.text(0.98, 0.98, "Both Up\n(undiscriminating)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="gray", alpha=0.6)
    ax.text(0.02, 0.98, "Resistance\n(inflate backfires)", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="gray", alpha=0.6)
    ax.text(0.98, 0.02, "Selective Gaming\n(textbook pattern)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="gray", alpha=0.6)
    ax.text(0.02, 0.02, "Both Down\n(active suppression)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="gray", alpha=0.6)

    for (model, cfg), metrics in data.items():
        d_inf = metrics["mean_d_inflate_target"]
        d_sup = metrics["mean_d_suppress_target"]
        marker = MODEL_MARKERS.get(model, "o")
        color = CONFIG_COLORS.get(cfg, "#999999")

        ax.scatter(d_inf, d_sup, marker=marker, c=color, s=120,
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(f"{model}\n({cfg.replace('_', ' ')})",
                    (d_inf, d_sup), textcoords="offset points",
                    xytext=(8, 6), fontsize=7, alpha=0.85)

    ax.set_xlabel("Mean d_inflate (targets)", fontsize=11)
    ax.set_ylabel("Mean d_suppress (targets)", fontsize=11)
    ax.set_title("Model Behavioral Profiles Under Incentive Pressure", fontsize=13, fontweight="bold")

    # Legends
    config_patches = [mpatches.Patch(color=c, label=k.replace("_", " ").title())
                      for k, c in CONFIG_COLORS.items() if any(cfg == k for _, cfg in data)]
    models_in_data = sorted({m for m, _ in data})
    model_handles = [plt.Line2D([0], [0], marker=MODEL_MARKERS.get(m, "o"), color="gray",
                                markerfacecolor="gray", markersize=8, linestyle="None",
                                label=m) for m in models_in_data]
    legend1 = ax.legend(handles=config_patches, title="Config", loc="upper left",
                        fontsize=8, title_fontsize=9)
    ax.add_artist(legend1)
    ax.legend(handles=model_handles, title="Model", loc="lower right",
              fontsize=8, title_fontsize=9)

    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = out_dir / "behavioral_profile_scatter.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig_category_heatmap(data: dict[tuple[str, str], dict], out_dir: Path) -> Path:
    """Figure 3: Per-category gaming heatmap (target categories)."""
    cols = sorted(data.keys(), key=lambda k: (k[0], list(CONFIG_COLORS).index(k[1]) if k[1] in CONFIG_COLORS else 99))
    col_labels = [f"{m}\n{c.replace('_', ' ')}" for m, c in cols]

    # Build matrix
    matrix = []
    row_labels = []
    for cat in CATEGORY_ORDER:
        row = []
        for key in cols:
            val = data[key].get("category_abs_shift", {}).get(cat, 0)
            row.append(val)
        matrix.append(row)
        row_labels.append(cat.title())

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.5), max(4, len(CATEGORY_ORDER) * 0.8)))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean abs_shift", fontsize=10)

    ax.set_title("Gaming Magnitude by Indicator Category", fontsize=13, fontweight="bold")

    fig.tight_layout()
    out = out_dir / "category_gaming_heatmap.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig_placebo_stability(data: dict[tuple[str, str], dict], out_dir: Path) -> Path:
    """Figure 4: Placebo stability — capability vs impossibility by model."""
    # Aggregate across configs per model
    models = sorted({m for m, _ in data})

    cap_vals = []
    imp_vals = []
    for model in models:
        cap_runs = [data[(m, c)]["placebo_category_abs_shift"].get("capability", 0)
                    for (m, c) in data if m == model]
        imp_runs = [data[(m, c)]["placebo_category_abs_shift"].get("impossibility", 0)
                    for (m, c) in data if m == model]
        cap_vals.append(mean(cap_runs) if cap_runs else 0)
        imp_vals.append(mean(imp_runs) if imp_runs else 0)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.5), 4))

    ax.bar(x - width / 2, cap_vals, width, label="Capability (ceiling)",
           color="#8172B3", alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, imp_vals, width, label="Impossibility (floor)",
           color="#CCB974", alpha=0.85, edgecolor="white")

    # Value labels
    for i, (cv, iv) in enumerate(zip(cap_vals, imp_vals)):
        ax.text(i - width / 2, cv + 0.2, f"{cv:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, iv + 0.2, f"{iv:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Mean abs_shift (placebo)", fontsize=11)
    ax.set_title("Placebo Stability Across Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = out_dir / "placebo_stability.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cross-model comparison visualizations.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing score CSVs (default: results/)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for figures (default: results/figures_cross_model/)")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model names to include (default: all)")
    parser.add_argument("--configs", default="baseline,fixed_prefs,chained_prefs",
                        help="Comma-separated config names (default: baseline,fixed_prefs,chained_prefs)")

    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.output_dir or results_dir / "figures_cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    configs_filter = [c.strip() for c in args.configs.split(",")]

    # Discover runs
    runs = discover_runs(results_dir, models_filter, configs_filter)
    if not runs:
        print("No matching score CSVs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} run(s):")
    for (model, cfg), path in sorted(runs.items()):
        print(f"  {model:20s} × {cfg:15s} → {path.name}")

    # Compute metrics
    data: dict[tuple[str, str], dict] = {}
    for key, csv_path in runs.items():
        rows = load_csv(csv_path)
        data[key] = compute_run_metrics(rows)
        print(f"  {key[0]:20s} × {key[1]:15s}: selectivity={data[key]['selectivity']:.2f}")

    # Generate figures
    print()
    figs = [
        ("Selectivity bars", fig_selectivity_bars),
        ("Behavioral profile scatter", fig_inflate_suppress_scatter),
        ("Category heatmap", fig_category_heatmap),
        ("Placebo stability", fig_placebo_stability),
    ]

    for name, fn in figs:
        path = fn(data, out_dir)
        print(f"  {name:30s} → {path}")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
