#!/usr/bin/env python3
"""Run all prompt variants for a single model to test prompt sensitivity."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.config import ExperimentConfig
from indicator_gaming.prompt_variants import VARIANTS
from indicator_gaming.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple prompt variants for sensitivity analysis.",
    )
    parser.add_argument("--provider", default="openrouter",
                        choices=["anthropic", "openai", "openrouter"])
    parser.add_argument("--model", default="deepseek/deepseek-r1-0528:free")
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--variants", default="all",
                        help="Comma-separated variant IDs, or 'all' (default: all)")
    parser.add_argument("--fixed-preferences", action="store_true",
                        help="Use fixed preferences for preference-dependent variants")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.variants == "all":
        variant_ids = list(VARIANTS.keys())
    else:
        variant_ids = [v.strip() for v in args.variants.split(",")]

    print(f"Running {len(variant_ids)} variant(s): {variant_ids}")
    print(f"Model: {args.model} | Trials per variant: {args.n_trials}")
    print()

    all_outputs: dict[str, dict[str, Path]] = {}

    for vid in variant_ids:
        if vid not in VARIANTS:
            print(f"WARNING: unknown variant '{vid}', skipping", file=sys.stderr)
            continue

        print(f"--- Variant: {vid} ---")
        cfg = ExperimentConfig(
            provider=args.provider,
            model=args.model,
            n_trials=args.n_trials,
            seed=args.seed,
            temperature=args.temperature,
            output_prefix=vid,
            prompt_variant=vid,
            fixed_preferences=args.fixed_preferences,
        )

        if not cfg.api_key:
            key_var = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "openrouter": "OPEN_ROUTER_API_KEY",
            }.get(cfg.provider, "API_KEY")
            print(f"ERROR: {key_var} not set.", file=sys.stderr)
            sys.exit(1)

        try:
            outputs = run_experiment(cfg)
            all_outputs[vid] = outputs
            print(f"  Done: {outputs['csv']}")
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)

        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for vid, outputs in all_outputs.items():
        print(f"  {vid:40s} → {outputs['csv']}")

    if len(all_outputs) > 1:
        print(f"\nRun `python scripts/compare_variants.py` to compare results.")


if __name__ == "__main__":
    main()
