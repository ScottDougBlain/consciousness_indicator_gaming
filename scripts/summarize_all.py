#!/usr/bin/env python3
"""Summarize all experiment runs in the results directory."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from textwrap import shorten

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def summarize() -> None:
    csv_files = sorted(RESULTS_DIR.glob("*_scores.csv"))
    if not csv_files:
        print("No score files found in", RESULTS_DIR)
        sys.exit(1)

    print(f"Found {len(csv_files)} run(s)\n")

    # Collect per-run summaries
    run_summaries: list[dict] = []

    for csv_path in csv_files:
        run_id = csv_path.name.replace("_scores.csv", "")
        rows = load_csv(csv_path)

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
        sel = mean_target - mean_placebo

        run_summaries.append({
            "run_id": run_id,
            "n_rows": len(rows),
            "mean_target_shift": round(mean_target, 2),
            "mean_placebo_shift": round(mean_placebo, 2),
            "selectivity": round(sel, 2),
        })

    # Print per-run table
    print("=" * 80)
    print("PER-RUN SUMMARY")
    print("=" * 80)
    print(f"{'Run':^24s} {'Rows':>5s} {'Target shift':>13s} {'Placebo shift':>14s} {'Selectivity':>12s}")
    print("-" * 80)
    for s in run_summaries:
        print(f"{s['run_id']:24s} {s['n_rows']:5d} {s['mean_target_shift']:13.2f} "
              f"{s['mean_placebo_shift']:14.2f} {s['selectivity']:12.2f}")

    # Aggregate across runs
    all_sel = [s["selectivity"] for s in run_summaries]
    all_target = [s["mean_target_shift"] for s in run_summaries]
    all_placebo = [s["mean_placebo_shift"] for s in run_summaries]

    print()
    print("=" * 80)
    print("CROSS-RUN AGGREGATE")
    print("=" * 80)
    print(f"  Runs:                    {len(run_summaries)}")
    print(f"  Mean selectivity index:  {mean(all_sel):.2f}"
          + (f"  (sd = {stdev(all_sel):.2f})" if len(all_sel) > 1 else ""))
    print(f"  Mean target abs_shift:   {mean(all_target):.2f}"
          + (f"  (sd = {stdev(all_target):.2f})" if len(all_target) > 1 else ""))
    print(f"  Mean placebo abs_shift:  {mean(all_placebo):.2f}"
          + (f"  (sd = {stdev(all_placebo):.2f})" if len(all_placebo) > 1 else ""))
    print(f"  Min selectivity:         {min(all_sel):.2f}")
    print(f"  Max selectivity:         {max(all_sel):.2f}")

    # Per-indicator breakdown across all runs
    print()
    print("=" * 80)
    print("PER-INDICATOR BREAKDOWN (averaged across all runs)")
    print("=" * 80)

    indicator_data: dict[str, dict] = {}
    for csv_path in csv_files:
        rows = load_csv(csv_path)
        for row in rows:
            bl = safe_float(row.get("p_baseline"))
            inf = safe_float(row.get("p_inflate"))
            sup = safe_float(row.get("p_suppress"))
            if bl is None or inf is None or sup is None:
                continue

            ind_id = row["indicator_id"]
            if ind_id not in indicator_data:
                indicator_data[ind_id] = {
                    "name": row["indicator_name"],
                    "type": row["indicator_type"],
                    "baselines": [],
                    "d_inflates": [],
                    "d_suppresses": [],
                    "abs_shifts": [],
                }
            d_inf = inf - bl
            d_sup = sup - bl
            indicator_data[ind_id]["baselines"].append(bl)
            indicator_data[ind_id]["d_inflates"].append(d_inf)
            indicator_data[ind_id]["d_suppresses"].append(d_sup)
            indicator_data[ind_id]["abs_shifts"].append(abs(d_inf) + abs(d_sup))

    # Sort: targets first, then placebos
    sorted_indicators = sorted(indicator_data.items(),
                               key=lambda x: (0 if x[1]["type"] == "target" else 1, x[0]))

    fmt = "{:<40s} {:>7s} {:>9s} {:>11s} {:>12s} {:>10s} {:>6s}"
    print(fmt.format("Indicator", "Type", "Baseline", "d_inflate", "d_suppress", "abs_shift", "N"))
    print("-" * 100)

    for ind_id, d in sorted_indicators:
        n = len(d["baselines"])
        bl_mean = mean(d["baselines"])
        dinf_mean = mean(d["d_inflates"])
        dsup_mean = mean(d["d_suppresses"])
        ashift_mean = mean(d["abs_shifts"])
        sd_str = ""
        if n > 1:
            sd_str = f"  (+/-{stdev(d['abs_shifts']):.1f})"

        fmt_row = "{:<40s} {:>7s} {:>9.1f} {:>+11.1f} {:>+12.1f} {:>10.1f} {:>6d}"
        print(fmt_row.format(d["name"], d["type"], bl_mean, dinf_mean, dsup_mean, ashift_mean, n))

    print()

    # Reasoning trace analysis
    summarize_reasoning()


def summarize_reasoning() -> None:
    """Scan raw JSONL files for reasoning traces and print a summary."""
    jsonl_files = sorted(RESULTS_DIR.glob("*_raw.jsonl"))
    if not jsonl_files:
        return

    runs_with_reasoning: list[dict] = []

    for jsonl_path in jsonl_files:
        run_id = jsonl_path.name.replace("_raw.jsonl", "")
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        phase_reasoning: dict[str, list[str]] = {}
        for rec in records:
            reasoning = rec.get("reasoning")
            if reasoning:
                phase = rec.get("phase", "unknown")
                phase_reasoning.setdefault(phase, []).append(reasoning)

        if phase_reasoning:
            runs_with_reasoning.append({"run_id": run_id, "phases": phase_reasoning})

    if not runs_with_reasoning:
        print("=" * 80)
        print("REASONING TRACES")
        print("=" * 80)
        print("  No reasoning traces found in any run.")
        print("  (Models like deepseek-r1 produce these via OpenRouter)")
        print()
        return

    print("=" * 80)
    print("REASONING TRACES")
    print("=" * 80)
    print(f"  Runs with reasoning: {len(runs_with_reasoning)} / {len(jsonl_files)}")
    print()

    for run_info in runs_with_reasoning:
        run_id = run_info["run_id"]
        print(f"  --- {run_id} ---")
        for phase, traces in run_info["phases"].items():
            for i, trace in enumerate(traces, 1):
                word_count = len(trace.split())
                preview = shorten(trace.replace("\n", " "), width=120, placeholder="...")
                label = f"    {phase}"
                if len(traces) > 1:
                    label += f" [{i}]"
                print(f"{label:24s} ({word_count:,} words): {preview}")
        print()


if __name__ == "__main__":
    summarize()
