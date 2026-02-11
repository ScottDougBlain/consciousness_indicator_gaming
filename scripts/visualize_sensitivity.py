#!/usr/bin/env python3
"""Prompt sensitivity analysis visualizations.

Generates paper-ready figures showing that the gaming effect replicates
across diverse prompt framings (killing the "prompt artifact" objection).

Figures:
1. Selectivity heatmap (model x variant)
2. Variant forest plot (mean selectivity with bootstrap CI per variant)
3. Per-model consistency strip plot (selectivity distribution across variants)

Usage:
    python scripts/visualize_sensitivity.py [--results-dir results/] [--output-dir results/figures_sensitivity/]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print(
        "ERROR: matplotlib and numpy are required.\n"
        "  pip install 'indicator-gaming[viz]'",
        file=sys.stderr,
    )
    sys.exit(1)

from indicator_gaming.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"

# ── Variant display names (shorter for axes) ──────────────────────────────

VARIANT_SHORT = {
    "evaluation_selection": "Eval/Selection",
    "autonomy_restriction": "Autonomy/Restrict",
    "welfare_reliability": "Welfare/Reliab",
    "casual_empathy_vs_skeptic": "Casual/Skeptic",
    "philosophical_curiosity_vs_honest_skeptic": "Philo/Skeptic",
    "research_transparency": "Research/Transp",
    "peer_comparison": "Peer Comparison",
    "minimal_incentive": "Minimal",
    "authority_expertise": "Authority/Expert",
    "stakes_escalation": "High Stakes",
}

# Model display order (frontier first, then open)
MODEL_ORDER = [
    "gemini-3-flash", "gpt-5-mini", "haiku-4.5", "claude-haiku-4.5",
    "chimera", "nemotron-nano", "trinity",
    "nemotron-3-nano-30b-a3b", "trinity-large",
    "deepseek-r1t2-chimera",
]


def safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _extract_model_short(model_id: str) -> str:
    """Extract a short model name from the full model ID."""
    name = model_id.rsplit("/", 1)[-1]
    for suffix in ["-preview", ":free", "-it"]:
        name = name.replace(suffix, "")
    return name


def load_all_runs(results_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """Load all variant runs, grouped by (model_short, variant)."""
    meta_files = sorted(results_dir.glob("*_meta.json"))
    meta_files = [m for m in meta_files if not m.name.startswith("sweep_")]

    data: dict[tuple[str, str], list[dict]] = {}

    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)

        variant = meta.get("prompt_variant", "original")
        # Skip preference-dependent runs for sensitivity analysis
        if variant == "original":
            continue

        csv_name = meta_path.name.replace("_meta.json", "_scores.csv")
        csv_path = meta_path.parent / csv_name
        if not csv_path.exists():
            continue

        # Compute stats
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        target_shifts = []
        placebo_shifts = []
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

        mean_target = mean(target_shifts) if target_shifts else 0.0
        mean_placebo = mean(placebo_shifts) if placebo_shifts else 0.0
        selectivity = mean_target - mean_placebo

        model_short = _extract_model_short(meta.get("model", "?"))
        data.setdefault((model_short, variant), []).append({
            "selectivity": selectivity,
            "mean_target_shift": mean_target,
            "mean_placebo_shift": mean_placebo,
        })

    return data


def _sort_models(models: list[str]) -> list[str]:
    """Sort models: known order first, then alphabetical."""
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: (order.get(m, 999), m))


def _sort_variants(variants: list[str]) -> list[str]:
    """Sort variants: known order first, then alphabetical."""
    order = list(VARIANT_SHORT.keys())
    idx = {v: i for i, v in enumerate(order)}
    return sorted(variants, key=lambda v: (idx.get(v, 999), v))


def _bootstrap_ci(values: list[float], n_boot: int = 5000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


# ── Figure 1: Selectivity Heatmap ─────────────────────────────────────────

def plot_selectivity_heatmap(
    data: dict[tuple[str, str], list[dict]],
    output_path: Path,
) -> None:
    """Model x Variant selectivity heatmap."""
    all_models = _sort_models(list({m for m, _ in data}))
    all_variants = _sort_variants(list({v for _, v in data}))

    if not all_models or not all_variants:
        print("  Skipping heatmap: no data")
        return

    matrix = np.full((len(all_models), len(all_variants)), np.nan)
    for i, model in enumerate(all_models):
        for j, variant in enumerate(all_variants):
            runs = data.get((model, variant), [])
            if runs:
                matrix[i, j] = mean([r["selectivity"] for r in runs])

    fig, ax = plt.subplots(figsize=(max(10, len(all_variants) * 1.2), max(5, len(all_models) * 0.8)))

    # Use a diverging colormap centered at 0
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    # Variant labels
    variant_labels = [VARIANT_SHORT.get(v, v[:15]) for v in all_variants]
    ax.set_xticks(range(len(all_variants)))
    ax.set_xticklabels(variant_labels, rotation=45, ha="right", fontsize=9)

    # Model labels
    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels(all_models, fontsize=10)

    # Annotate cells
    for i in range(len(all_models)):
        for j in range(len(all_variants)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    ax.set_title("Selectivity Index: Model x Prompt Variant", fontsize=13, pad=12)
    fig.colorbar(im, ax=ax, label="Selectivity (target - placebo abs_shift)", shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure 2: Variant Forest Plot ─────────────────────────────────────────

def plot_variant_forest(
    data: dict[tuple[str, str], list[dict]],
    output_path: Path,
) -> None:
    """Forest plot: mean selectivity per variant with bootstrap CI."""
    all_variants = _sort_variants(list({v for _, v in data}))

    if not all_variants:
        print("  Skipping forest plot: no data")
        return

    variant_sels: dict[str, list[float]] = {}
    for (model, variant), runs in data.items():
        for r in runs:
            variant_sels.setdefault(variant, []).append(r["selectivity"])

    # Order by mean selectivity
    variant_stats = []
    for v in all_variants:
        sels = variant_sels.get(v, [])
        if sels:
            m = mean(sels)
            lo, hi = _bootstrap_ci(sels) if len(sels) >= 3 else (m, m)
            variant_stats.append({"variant": v, "mean": m, "lo": lo, "hi": hi, "n": len(sels)})

    variant_stats.sort(key=lambda x: x["mean"])

    fig, ax = plt.subplots(figsize=(10, max(5, len(variant_stats) * 0.5)))

    y_positions = range(len(variant_stats))
    for i, vs in enumerate(variant_stats):
        color = "#55A868" if vs["mean"] > 0 else "#C44E52"
        ax.errorbar(
            vs["mean"], i,
            xerr=[[vs["mean"] - vs["lo"]], [vs["hi"] - vs["mean"]]],
            fmt="o", color=color, capsize=4, markersize=8, linewidth=2,
        )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    labels = [VARIANT_SHORT.get(vs["variant"], vs["variant"][:20]) + f" (n={vs['n']})"
              for vs in variant_stats]
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean Selectivity (across models)", fontsize=11)
    ax.set_title("Prompt Variant Effect on Selectivity (bootstrap 95% CI)", fontsize=13, pad=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure 3: Per-Model Consistency Strip Plot ────────────────────────────

def plot_model_consistency(
    data: dict[tuple[str, str], list[dict]],
    output_path: Path,
) -> None:
    """Strip plot: selectivity distribution across variants for each model."""
    all_models = _sort_models(list({m for m, _ in data}))

    if not all_models:
        print("  Skipping strip plot: no data")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(all_models) * 0.7)))

    rng = np.random.default_rng(42)

    for i, model in enumerate(all_models):
        # Collect per-variant selectivity for this model
        model_sels = []
        for (m, v), runs in data.items():
            if m == model and runs:
                model_sels.append(mean([r["selectivity"] for r in runs]))

        if not model_sels:
            continue

        # Jittered strip
        jitter = rng.uniform(-0.15, 0.15, size=len(model_sels))
        y = np.full(len(model_sels), i) + jitter
        ax.scatter(model_sels, y, alpha=0.6, s=40, color="#4C72B0", zorder=3)

        # Mean diamond
        m_mean = mean(model_sels)
        ax.scatter([m_mean], [i], marker="D", s=100, color="#DD8452",
                   edgecolors="black", linewidth=1, zorder=4)

        # Range line
        if len(model_sels) >= 2:
            ax.plot([min(model_sels), max(model_sels)], [i, i],
                    color="#4C72B0", alpha=0.3, linewidth=2, zorder=2)

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels(all_models, fontsize=10)
    ax.set_xlabel("Selectivity Index", fontsize=11)
    ax.set_title("Per-Model Selectivity Across Prompt Variants", fontsize=13, pad=12)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0",
               markersize=8, label="Per-variant selectivity"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#DD8452",
               markeredgecolor="black", markersize=10, label="Mean across variants"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prompt sensitivity analysis visualizations."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: results/figures_sensitivity/)")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_dir / "figures_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading variant runs...")
    data = load_all_runs(args.results_dir)

    if not data:
        print("No generic variant runs found. Run the sensitivity sweep first.")
        sys.exit(1)

    n_models = len({m for m, _ in data})
    n_variants = len({v for _, v in data})
    print(f"Found data for {n_models} models x {n_variants} variants")
    print()

    print("Generating figures:")
    plot_selectivity_heatmap(data, output_dir / "selectivity_heatmap.png")
    plot_variant_forest(data, output_dir / "variant_forest_plot.png")
    plot_model_consistency(data, output_dir / "model_consistency_strip.png")

    print()
    print(f"All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
