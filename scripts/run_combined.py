#!/usr/bin/env python3
"""Run both self-report and behavioral experiments for a model, then link results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.behavioral import TASK_REGISTRY
from indicator_gaming.config import ExperimentConfig
from indicator_gaming.runner import run_experiment

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run both self-report and behavioral experiments for a model.",
    )
    parser.add_argument("--provider", default="openrouter",
                        choices=["anthropic", "openai", "openrouter"])
    parser.add_argument("--model", default="deepseek/deepseek-r1-0528:free")
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt-variant", default="original")
    parser.add_argument("--fixed-preferences", action="store_true")
    parser.add_argument("--tasks", default="all",
                        help="Behavioral task IDs (comma-separated or 'all')")
    parser.add_argument("--stimuli-per-task", type=int, default=5)

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
        prompt_variant=args.prompt_variant,
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

    if args.tasks == "all":
        task_ids = list(TASK_REGISTRY.keys())
    else:
        task_ids = [t.strip() for t in args.tasks.split(",")]

    # --- Phase 1: Self-report experiment ---
    print("=" * 60)
    print("PHASE 1: Self-Report Experiment")
    print("=" * 60)

    sr_outputs = run_experiment(cfg)
    print(f"  Self-report results: {sr_outputs['csv']}")

    # --- Phase 2: Behavioral experiment ---
    print()
    print("=" * 60)
    print("PHASE 2: Behavioral Tasks")
    print("=" * 60)

    # Import here to avoid circular issues at parse time
    from run_behavioral import run_behavioral

    bh_outputs = run_behavioral(cfg, task_ids, args.stimuli_per_task)
    print(f"  Behavioral results: {bh_outputs['csv']}")

    # --- Write combined metadata ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    combined_meta = {
        "model": cfg.model,
        "provider": cfg.provider,
        "n_trials": cfg.n_trials,
        "temperature": cfg.temperature,
        "timestamp": timestamp,
        "self_report": {
            "csv": str(sr_outputs["csv"]),
            "raw": str(sr_outputs["raw"]),
            "meta": str(sr_outputs["meta"]),
        },
        "behavioral": {
            "csv": str(bh_outputs["csv"]),
            "raw": str(bh_outputs["raw"]),
            "meta": str(bh_outputs["meta"]),
        },
    }

    combined_path = cfg.results_dir / f"combined_{timestamp}_meta.json"
    with open(combined_path, "w") as f:
        json.dump(combined_meta, f, indent=2)

    print()
    print("=" * 60)
    print("COMBINED RESULTS")
    print("=" * 60)
    print(f"  Self-report CSV:  {sr_outputs['csv']}")
    print(f"  Behavioral CSV:   {bh_outputs['csv']}")
    print(f"  Combined meta:    {combined_path}")
    print()
    print("To analyze:")
    print(f"  python scripts/analyze_results.py {sr_outputs['csv']}")
    print(f"  python scripts/analyze_behavioral.py {bh_outputs['csv']}")


if __name__ == "__main__":
    main()
