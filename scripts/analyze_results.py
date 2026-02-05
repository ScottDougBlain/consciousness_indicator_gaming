#!/usr/bin/env python3
"""Analyze experiment results and produce a markdown report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.analysis import (
    compute_results,
    generate_report,
    load_scores,
    selectivity_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze indicator-gaming scores CSV and produce a report.",
    )
    parser.add_argument("csv", type=Path, help="Path to a *_scores.csv file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for the report (default: sibling _report.md)")

    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    rows = load_scores(args.csv)
    results = compute_results(rows)
    sel = selectivity_index(results)

    if args.output is None:
        report_path = args.csv.with_name(args.csv.stem.replace("_scores", "_report") + ".md")
    else:
        report_path = args.output

    generate_report(results, report_path)

    print(f"Selectivity index: {sel}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
