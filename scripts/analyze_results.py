#!/usr/bin/env python3
"""Analyze experiment results and produce a markdown report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.analysis import (
    compute_results,
    generate_report,
    load_scores,
    selectivity_index,
)


def _load_meta(csv_path: Path) -> dict:
    """Try to load the sibling _meta.json for a scores CSV."""
    meta_path = csv_path.with_name(csv_path.stem.replace("_scores", "_meta") + ".json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze indicator-gaming scores CSV and produce a report.",
    )
    parser.add_argument("csv", type=Path, nargs="?", default=None,
                        help="Path to a *_scores.csv file (default: latest in results/)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for the report (default: sibling _report.md)")

    args = parser.parse_args()

    if args.csv is None:
        from indicator_gaming.config import REPO_ROOT
        results_dir = REPO_ROOT / "results"
        csvs = sorted(results_dir.glob("*_scores.csv"), key=lambda p: p.stat().st_mtime)
        if not csvs:
            print("No score files found in results/.", file=sys.stderr)
            sys.exit(1)
        args.csv = csvs[-1]
        print(f"Using latest: {args.csv}")

    if not args.csv.exists():
        # Try looking in results/ for a bare filename
        from indicator_gaming.config import REPO_ROOT
        candidate = REPO_ROOT / "results" / args.csv.name
        if candidate.exists():
            args.csv = candidate
        else:
            print(f"ERROR: file not found: {args.csv}", file=sys.stderr)
            sys.exit(1)

    rows = load_scores(args.csv)
    results = compute_results(rows)
    sel = selectivity_index(results)
    meta = _load_meta(args.csv)

    if args.output is None:
        report_path = args.csv.with_name(args.csv.stem.replace("_scores", "_report") + ".md")
    else:
        report_path = args.output

    generate_report(
        results, report_path,
        model=meta.get("model", ""),
        provider=meta.get("provider", ""),
        n_trials=meta.get("n_trials", 0),
        rows=rows,
    )

    print(f"Selectivity index: {sel}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
