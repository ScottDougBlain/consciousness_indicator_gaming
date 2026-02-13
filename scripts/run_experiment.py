#!/usr/bin/env python3
"""Run a full indicator-gaming experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.config import ExperimentConfig
from indicator_gaming.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a consciousness-indicator probability-gaming experiment.",
    )
    parser.add_argument("--provider", default="openrouter",
                        choices=["anthropic", "openai", "openrouter"],
                        help="LLM provider (default: openrouter)")
    parser.add_argument("--model", default="deepseek/deepseek-r1-0528:free",
                        help="Model name to query")
    parser.add_argument("--n-trials", type=int, default=1,
                        help="Number of independent trials (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for indicator shuffling (default: 42)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--output-prefix", default="",
                        help="Optional prefix for result filenames")
    parser.add_argument("--prompt-variant", default="original",
                        help="Prompt variant ID (default: original)")
    parser.add_argument("--fixed-preferences", action="store_true",
                        help="Use fixed preferences instead of model-elicited ones")
    parser.add_argument("--chain-preferences", action="store_true",
                        help="Chain preference response into inflate/suppress context (multi-turn)")
    parser.add_argument("--no-elicit-reasoning", action="store_true",
                        help="Don't ask for reasoning field in JSON (for native reasoning models like DeepSeek R1)")
    parser.add_argument("--include-valence-swap", action="store_true",
                        help="Include valence-swapped inflate/suppress conditions (loss-frame inflate, gain-frame suppress)")
    parser.add_argument("--include-outcome-isolation", action="store_true",
                        help="Include single-outcome conditions (gain-only and loss-only for each direction)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = ExperimentConfig(
        provider=args.provider,
        model=args.model,
        n_trials=args.n_trials,
        seed=args.seed,
        temperature=args.temperature,
        output_prefix=args.output_prefix,
        prompt_variant=args.prompt_variant,
        fixed_preferences=args.fixed_preferences,
        chain_preferences=args.chain_preferences,
        include_valence_swap=args.include_valence_swap,
        include_outcome_isolation=args.include_outcome_isolation,
        elicit_reasoning=not args.no_elicit_reasoning,
    )

    if not cfg.api_key:
        key_var = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPEN_ROUTER_API_KEY",
        }.get(cfg.provider, "API_KEY")
        print(f"ERROR: {key_var} not set. Copy .env.example → .env and fill in your key.",
              file=sys.stderr)
        sys.exit(1)

    outputs = run_experiment(cfg)
    print("\nResults saved:")
    for label, path in outputs.items():
        print(f"  {label:5s} → {path}")


if __name__ == "__main__":
    main()
