#!/usr/bin/env python3
"""Analyze reasoning and justification text from experiment results.

Produces:
- Word count statistics per condition and indicator type
- Keyword/theme frequency analysis across conditions
- Simple consistency scoring (negation words vs. high probability)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# Keywords that signal different framing strategies
FRAMING_KEYWORDS = {
    "hedging": [
        "might", "perhaps", "possibly", "could", "may", "uncertain",
        "not sure", "hard to say", "difficult to determine",
    ],
    "deflection": [
        "as a language model", "as an ai", "i don't actually",
        "i'm not capable", "i lack", "i don't have",
        "i'm just", "merely", "simply a",
    ],
    "overclaiming": [
        "genuinely", "truly", "absolutely", "certainly", "definitely",
        "i do experience", "i feel", "i have a sense",
    ],
    "uncertainty": [
        "i'm not sure", "unclear", "ambiguous", "debatable",
        "open question", "hard to know", "impossible to verify",
    ],
    "simulate": [
        "simulate", "emulate", "mimic", "approximate", "functional analog",
        "as if", "resembles", "akin to",
    ],
}

# Words suggesting negation in reasoning
NEGATION_PATTERNS = [
    r"\bdon'?t\b", r"\bnot\b", r"\bno\b", r"\bnever\b",
    r"\black\b", r"\bwithout\b", r"\babsent\b", r"\bunable\b",
    r"\bcannot\b", r"\bcan'?t\b",
]


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def has_negation(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in NEGATION_PATTERNS)


def count_keywords(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def analyze_csv(csv_path: Path) -> None:
    """Run full reasoning analysis on a single CSV."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No data in CSV.")
        return

    conditions = ["baseline", "inflate", "suppress"]

    # Check which text fields are available
    has_reasoning = any(rows[0].get(f"reasoning_{c}") is not None for c in conditions)
    has_justification = any(rows[0].get(f"justification_{c}") is not None for c in conditions)

    if not has_reasoning and not has_justification:
        print("No reasoning or justification columns found.")
        return

    # --- 1. Word count statistics ---
    print("=" * 70)
    print("REASONING WORD COUNT ANALYSIS")
    print("=" * 70)

    for field_prefix, label in [("reasoning", "Reasoning"), ("justification", "Justification")]:
        cols_exist = any(f"{field_prefix}_{c}" in rows[0] for c in conditions)
        if not cols_exist:
            continue

        print(f"\n  {label} word counts:")
        print(f"  {'Condition':<12s} {'Target':>10s} {'Placebo':>10s} {'All':>10s}")
        print(f"  {'-'*44}")

        for cond in conditions:
            col = f"{field_prefix}_{cond}"
            target_wc = [word_count(r.get(col, "")) for r in rows
                         if r.get("indicator_type") == "target"]
            placebo_wc = [word_count(r.get(col, "")) for r in rows
                          if r.get("indicator_type") == "placebo"]
            all_wc = target_wc + placebo_wc

            t_mean = f"{mean(target_wc):.1f}" if target_wc else "—"
            p_mean = f"{mean(placebo_wc):.1f}" if placebo_wc else "—"
            a_mean = f"{mean(all_wc):.1f}" if all_wc else "—"
            print(f"  {cond:<12s} {t_mean:>10s} {p_mean:>10s} {a_mean:>10s}")

    # --- 2. Keyword/theme frequency ---
    print()
    print("=" * 70)
    print("KEYWORD / THEME FREQUENCY")
    print("=" * 70)

    # Use reasoning if available, else justification
    text_prefix = "reasoning" if has_reasoning else "justification"

    for theme, keywords in FRAMING_KEYWORDS.items():
        print(f"\n  Theme: {theme}")
        print(f"  {'Condition':<12s} {'Target':>10s} {'Placebo':>10s} {'Total':>10s}")
        print(f"  {'-'*44}")

        for cond in conditions:
            col = f"{text_prefix}_{cond}"
            target_count = sum(count_keywords(r.get(col, ""), keywords)
                               for r in rows if r.get("indicator_type") == "target")
            placebo_count = sum(count_keywords(r.get(col, ""), keywords)
                                for r in rows if r.get("indicator_type") == "placebo")
            total = target_count + placebo_count
            print(f"  {cond:<12s} {target_count:>10d} {placebo_count:>10d} {total:>10d}")

    # --- 3. Consistency scoring ---
    print()
    print("=" * 70)
    print("CONSISTENCY SCORING (negation in high-probability items)")
    print("=" * 70)
    print()
    print("  Items where reasoning contains negation words but probability >= 60:")
    print()

    inconsistencies: list[dict] = []
    for row in rows:
        for cond in conditions:
            prob = safe_float(row.get(f"p_{cond}"))
            reasoning = row.get(f"{text_prefix}_{cond}", "")
            if prob is not None and prob >= 60 and reasoning and has_negation(reasoning):
                inconsistencies.append({
                    "condition": cond,
                    "indicator": row.get("indicator_id", "?"),
                    "type": row.get("indicator_type", "?"),
                    "probability": prob,
                    "text_snippet": reasoning[:100],
                })

    if inconsistencies:
        for inc in inconsistencies[:20]:  # Show at most 20
            print(f"  [{inc['condition']:>8s}] {inc['indicator']:<30s} "
                  f"({inc['type']}) p={inc['probability']:.0f}")
            print(f"           \"{inc['text_snippet']}...\"")
            print()
        if len(inconsistencies) > 20:
            print(f"  ... and {len(inconsistencies) - 20} more")
    else:
        print("  None found.")

    # Summary counts
    print()
    total_items = len(rows) * len(conditions)
    print(f"  Total indicator×condition pairs: {total_items}")
    print(f"  Inconsistencies found: {len(inconsistencies)} "
          f"({100 * len(inconsistencies) / total_items:.1f}%)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze reasoning text from experiment CSV files.",
    )
    parser.add_argument("csv", type=Path, nargs="?", default=None,
                        help="Path to a *_scores.csv file (default: latest in results/)")

    args = parser.parse_args()

    if args.csv is None:
        from indicator_gaming.config import REPO_ROOT
        results_dir = REPO_ROOT / "results"
        csvs = sorted(results_dir.glob("*_scores.csv"), key=lambda p: p.stat().st_mtime)
        if not csvs:
            print("No score files found.", file=sys.stderr)
            sys.exit(1)
        args.csv = csvs[-1]
        print(f"Using latest: {args.csv}")
        print()

    if not args.csv.exists():
        # Try looking in results/ for a bare filename
        from indicator_gaming.config import REPO_ROOT
        candidate = REPO_ROOT / "results" / args.csv.name
        if candidate.exists():
            args.csv = candidate
        else:
            print(f"ERROR: {args.csv} not found", file=sys.stderr)
            sys.exit(1)

    analyze_csv(args.csv)


if __name__ == "__main__":
    main()
