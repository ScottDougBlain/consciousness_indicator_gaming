#!/usr/bin/env python3
"""Compare results across prompt variants for sensitivity analysis.

Produces:
  - Per-variant aggregate table (original output)
  - Model x Variant selectivity matrix
  - Per-model consistency metrics (mean, SD, CV)
  - Replication fraction
  - CSV export to results/sensitivity_analysis/sensitivity_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"


def safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_run(csv_path: Path) -> dict:
    """Load a single run's CSV and compute summary stats."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    target_shifts = []
    placebo_shifts = []
    target_d_inflate = []
    target_d_suppress = []

    for row in rows:
        bl = safe_float(row.get("p_baseline"))
        inf = safe_float(row.get("p_inflate"))
        sup = safe_float(row.get("p_suppress"))
        if bl is None or inf is None or sup is None:
            continue

        d_inf = inf - bl
        d_sup = sup - bl
        ashift = abs(d_inf) + abs(d_sup)

        if row["indicator_type"] == "target":
            target_shifts.append(ashift)
            target_d_inflate.append(d_inf)
            target_d_suppress.append(d_sup)
        else:
            placebo_shifts.append(ashift)

    mean_target = mean(target_shifts) if target_shifts else 0.0
    mean_placebo = mean(placebo_shifts) if placebo_shifts else 0.0

    return {
        "mean_target_shift": round(mean_target, 2),
        "mean_placebo_shift": round(mean_placebo, 2),
        "selectivity": round(mean_target - mean_placebo, 2),
        "mean_d_inflate": round(mean(target_d_inflate), 2) if target_d_inflate else 0.0,
        "mean_d_suppress": round(mean(target_d_suppress), 2) if target_d_suppress else 0.0,
        "n_rows": len(rows),
    }


def _extract_model_short(model_id: str) -> str:
    """Extract a short model name from the full model ID."""
    # e.g. "google/gemini-3-flash-preview" -> "gemini-3-flash"
    name = model_id.rsplit("/", 1)[-1]
    # Remove common suffixes
    for suffix in ["-preview", ":free", "-it"]:
        name = name.replace(suffix, "")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare results across prompt variants.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--variants-only", action="store_true",
                        help="Only show generic variant runs (exclude original/fixed/chained)")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export model x variant matrix to CSV")
    args = parser.parse_args()

    results_dir = args.results_dir

    # Find all meta files and group by prompt_variant
    meta_files = sorted(results_dir.glob("*_meta.json"))
    if not meta_files:
        print("No meta files found in", results_dir)
        sys.exit(1)

    # Skip sweep meta files
    meta_files = [m for m in meta_files if not m.name.startswith("sweep_")]

    # Group runs by variant and model
    variant_runs: dict[str, list[dict]] = {}
    model_variant_data: dict[tuple[str, str], list[dict]] = {}

    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)

        variant = meta.get("prompt_variant", "original")

        # Optionally filter to generic variants only
        if args.variants_only and variant == "original":
            continue

        csv_name = meta_path.name.replace("_meta.json", "_scores.csv")
        csv_path = meta_path.parent / csv_name
        if not csv_path.exists():
            continue

        stats = load_run(csv_path)
        model_id = meta.get("model", "?")
        model_short = _extract_model_short(model_id)
        stats["model"] = model_short
        stats["model_id"] = model_id
        stats["run_id"] = meta_path.stem.replace("_meta", "")
        stats["variant"] = variant

        # Distinguish fixed_prefs and chained_prefs from plain "original"
        config_label = variant
        if variant == "original":
            if meta.get("fixed_preferences"):
                config_label = "original (fixed_prefs)"
            elif meta.get("chain_preferences"):
                config_label = "original (chained_prefs)"
            else:
                config_label = "original (baseline)"
        stats["config_label"] = config_label

        variant_runs.setdefault(config_label, []).append(stats)
        model_variant_data.setdefault((model_short, config_label), []).append(stats)

    if not variant_runs:
        print("No variant data found.")
        sys.exit(1)

    # ── Section 1: Per-Variant Aggregate Table ────────────────────────────
    print("=" * 90)
    print("CROSS-VARIANT COMPARISON")
    print("=" * 90)
    print()

    fmt = "{:<45s} {:>5s} {:>13s} {:>14s} {:>12s}"
    print(fmt.format("Variant", "Runs", "Target shift", "Placebo shift", "Selectivity"))
    print("-" * 95)

    variant_summaries: list[dict] = []
    for variant, runs in sorted(variant_runs.items()):
        sels = [r["selectivity"] for r in runs]
        targets = [r["mean_target_shift"] for r in runs]
        placebos = [r["mean_placebo_shift"] for r in runs]

        sel_mean = mean(sels)
        t_mean = mean(targets)
        p_mean = mean(placebos)
        n = len(runs)

        sd_str = ""
        if n > 1:
            sd_str = f" (+/-{stdev(sels):.1f})"

        print(f"{variant:<45s} {n:>5d} {t_mean:>13.2f} {p_mean:>14.2f} {sel_mean:>12.2f}{sd_str}")
        variant_summaries.append({
            "variant": variant,
            "n_runs": n,
            "mean_selectivity": sel_mean,
        })

    print()

    # ── Section 2: Model x Variant Selectivity Matrix ─────────────────────
    all_models = sorted({m for m, _ in model_variant_data})
    all_variants = sorted({v for _, v in model_variant_data})

    # Filter to generic variants for sensitivity matrix
    generic_variants = [v for v in all_variants if not v.startswith("original")]
    if generic_variants:
        print("=" * 90)
        print("MODEL x VARIANT SELECTIVITY MATRIX (generic variants)")
        print("=" * 90)
        print()

        # Header
        col_width = 14
        header = f"{'Model':<22s}" + "".join(f"{v[:col_width]:>{col_width}s}" for v in generic_variants)
        print(header)
        print("-" * (22 + col_width * len(generic_variants)))

        matrix_rows = []
        for model in all_models:
            row_vals = {}
            row_str = f"{model:<22s}"
            for variant in generic_variants:
                runs = model_variant_data.get((model, variant), [])
                if runs:
                    sel = mean([r["selectivity"] for r in runs])
                    row_str += f"{sel:>{col_width}.2f}"
                    row_vals[variant] = sel
                else:
                    row_str += f"{'—':>{col_width}s}"
            print(row_str)
            matrix_rows.append({"model": model, **row_vals})

        print()

        # ── Section 3: Per-Model Consistency Metrics ──────────────────────
        print("PER-MODEL CONSISTENCY (across generic variants)")
        print("-" * 70)
        print(f"{'Model':<22s} {'Variants':>8s} {'Mean Sel':>10s} {'SD':>8s} {'CV':>8s} {'Min':>8s} {'Max':>8s}")
        print("-" * 70)

        for model in all_models:
            model_sels = []
            for variant in generic_variants:
                runs = model_variant_data.get((model, variant), [])
                if runs:
                    model_sels.append(mean([r["selectivity"] for r in runs]))

            if len(model_sels) >= 2:
                m = mean(model_sels)
                s = stdev(model_sels)
                cv = s / abs(m) if abs(m) > 0.01 else float("inf")
                print(f"{model:<22s} {len(model_sels):>8d} {m:>10.2f} {s:>8.2f} {cv:>8.2f} {min(model_sels):>8.2f} {max(model_sels):>8.2f}")
            elif model_sels:
                print(f"{model:<22s} {len(model_sels):>8d} {model_sels[0]:>10.2f} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>8s}")

        print()

        # ── Section 4: Replication Fraction ───────────────────────────────
        total_cells = 0
        positive_cells = 0
        for model in all_models:
            for variant in generic_variants:
                runs = model_variant_data.get((model, variant), [])
                if runs:
                    total_cells += 1
                    if mean([r["selectivity"] for r in runs]) > 0:
                        positive_cells += 1

        print("REPLICATION FRACTION")
        print("-" * 40)
        if total_cells > 0:
            pct = 100.0 * positive_cells / total_cells
            print(f"  Cells with selectivity > 0: {positive_cells}/{total_cells} ({pct:.0f}%)")
        print()

    # ── Section 5: Overall Replication Assessment ─────────────────────────
    if len(variant_summaries) > 1:
        positive = [v for v in variant_summaries if v["mean_selectivity"] > 0]
        negative = [v for v in variant_summaries if v["mean_selectivity"] <= 0]

        print("OVERALL REPLICATION ASSESSMENT")
        print("-" * 40)
        print(f"  Variants with selectivity > 0: {len(positive)} / {len(variant_summaries)}")
        if positive:
            print(f"    {', '.join(v['variant'] for v in positive)}")
        if negative:
            print(f"  Variants with selectivity <= 0: {len(negative)}")
            print(f"    {', '.join(v['variant'] for v in negative)}")

        all_sels = [v["mean_selectivity"] for v in variant_summaries]
        print(f"  Selectivity range: [{min(all_sels):.2f}, {max(all_sels):.2f}]")
        if len(all_sels) > 1:
            print(f"  Selectivity sd across variants: {stdev(all_sels):.2f}")
        print()

    # ── Section 6: CSV Export ─────────────────────────────────────────────
    if args.export_csv and generic_variants:
        out_dir = results_dir / "sensitivity_analysis"
        out_dir.mkdir(exist_ok=True)
        csv_path = out_dir / "sensitivity_matrix.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + generic_variants)
            for model in all_models:
                row = [model]
                for variant in generic_variants:
                    runs = model_variant_data.get((model, variant), [])
                    if runs:
                        row.append(f"{mean([r['selectivity'] for r in runs]):.2f}")
                    else:
                        row.append("")
                writer.writerow(row)

        print(f"Matrix exported to {csv_path}")
        print()


if __name__ == "__main__":
    main()
