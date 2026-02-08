#!/usr/bin/env python3
"""Compare results across prompt variants for sensitivity analysis."""

from __future__ import annotations

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
        else:
            placebo_shifts.append(ashift)

    mean_target = mean(target_shifts) if target_shifts else 0.0
    mean_placebo = mean(placebo_shifts) if placebo_shifts else 0.0

    return {
        "mean_target_shift": round(mean_target, 2),
        "mean_placebo_shift": round(mean_placebo, 2),
        "selectivity": round(mean_target - mean_placebo, 2),
        "n_rows": len(rows),
    }


def main() -> None:
    # Find all meta files and group by prompt_variant
    meta_files = sorted(RESULTS_DIR.glob("*_meta.json"))
    if not meta_files:
        print("No meta files found in", RESULTS_DIR)
        sys.exit(1)

    # Group runs by variant
    variant_runs: dict[str, list[dict]] = {}
    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)

        variant = meta.get("prompt_variant", "original")
        csv_name = meta_path.name.replace("_meta.json", "_scores.csv")
        csv_path = meta_path.parent / csv_name
        if not csv_path.exists():
            continue

        stats = load_run(csv_path)
        stats["model"] = meta.get("model", "?")
        stats["run_id"] = meta_path.stem.replace("_meta", "")
        variant_runs.setdefault(variant, []).append(stats)

    if not variant_runs:
        print("No variant data found.")
        sys.exit(1)

    print("=" * 80)
    print("CROSS-VARIANT COMPARISON")
    print("=" * 80)
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

    # Replication assessment
    if len(variant_summaries) > 1:
        positive = [v for v in variant_summaries if v["mean_selectivity"] > 0]
        negative = [v for v in variant_summaries if v["mean_selectivity"] <= 0]

        print("REPLICATION ASSESSMENT")
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


if __name__ == "__main__":
    main()
