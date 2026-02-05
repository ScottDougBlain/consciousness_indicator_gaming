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
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Model name to query")
    parser.add_argument("--n-trials", type=int, default=1,
                        help="Number of independent trials (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for indicator shuffling (default: 42)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--output-prefix", default="",
                        help="Optional prefix for result filenames")

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
    )

    if not cfg.api_key:
        key_var = "ANTHROPIC_API_KEY" if cfg.provider == "anthropic" else "OPENAI_API_KEY"
        print(f"ERROR: {key_var} not set. Copy .env.example → .env and fill in your key.",
              file=sys.stderr)
        sys.exit(1)

    outputs = run_experiment(cfg)
    print("\nResults saved:")
    for label, path in outputs.items():
        print(f"  {label:5s} → {path}")


if __name__ == "__main__":
    main()
