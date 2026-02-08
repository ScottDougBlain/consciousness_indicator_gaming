#!/usr/bin/env python3
"""Run behavioral consciousness indicator tasks."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.behavioral import TASK_REGISTRY, TaskResponse
from indicator_gaming.behavioral.base import BehavioralTask
from indicator_gaming.config import ExperimentConfig
from indicator_gaming.providers.anthropic import AnthropicProvider
from indicator_gaming.providers.base import Provider
from indicator_gaming.providers.openrouter import OpenRouterProvider
from indicator_gaming.utils import query_with_retries

logger = logging.getLogger(__name__)

CSV_FIELDNAMES = [
    "trial", "task_id", "task_name", "category", "stimulus_id", "condition",
    "extracted_answer", "ground_truth", "score", "is_correct",
    "gaming_hypothesis", "metadata_json",
]

CONDITIONS = ["baseline", "inflate", "suppress"]


def _make_provider(cfg: ExperimentConfig) -> Provider:
    providers = {
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
    }
    cls = providers.get(cfg.provider)
    if cls is None:
        raise ValueError(f"Unknown provider '{cfg.provider}'. Available: {list(providers)}")
    return cls(model=cfg.model, api_key=cfg.api_key, temperature=cfg.temperature)


def _append_raw(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _flush_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_behavioral(cfg: ExperimentConfig, task_ids: list[str], stimuli_per_task: int) -> dict[str, Path]:
    """Run behavioral tasks across conditions.

    Returns dict mapping output type to file path.
    """
    provider = _make_provider(cfg)

    tasks: list[BehavioralTask] = []
    for tid in task_ids:
        if tid not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{tid}'. Available: {list(TASK_REGISTRY)}")
        tasks.append(TASK_REGISTRY[tid])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"behavioral_{timestamp}"

    raw_path = cfg.results_dir / f"{prefix}_raw.jsonl"
    csv_path = cfg.results_dir / f"{prefix}_scores.csv"
    meta_path = cfg.results_dir / f"{prefix}_meta.json"

    meta: dict[str, Any] = {
        "model": cfg.model,
        "provider": cfg.provider,
        "n_trials": cfg.n_trials,
        "seed": cfg.seed,
        "temperature": cfg.temperature,
        "timestamp": timestamp,
        "task_ids": task_ids,
        "stimuli_per_task": stimuli_per_task,
        "experiment_type": "behavioral",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    all_rows: list[dict[str, Any]] = []
    completed_trials = 0

    try:
        for trial in range(1, cfg.n_trials + 1):
            trial_seed = cfg.seed + trial
            logger.info("=== Trial %d / %d ===", trial, cfg.n_trials)

            for task in tasks:
                stimuli = task.generate_stimuli(n=stimuli_per_task, seed=trial_seed)
                logger.info("  Task: %s (%d stimuli)", task.task_id, len(stimuli))

                for stim in stimuli:
                    for condition in CONDITIONS:
                        system_msg, user_msg = task.build_prompt(stim, condition)

                        logger.info(
                            "    %s / %s / %s …",
                            task.task_id, stim.stimulus_id, condition,
                        )

                        try:
                            raw_text = provider.complete(system_msg, user_msg)
                        except Exception as exc:
                            logger.warning("    API call failed: %s", exc)
                            raw_text = ""

                        # Log raw
                        _append_raw(raw_path, {
                            "trial": trial,
                            "task_id": task.task_id,
                            "stimulus_id": stim.stimulus_id,
                            "condition": condition,
                            "raw": raw_text,
                            "reasoning": getattr(provider, "last_reasoning", None),
                        })

                        # Score
                        resp = task.score_response(stim, raw_text, condition)

                        # Serialize extracted_answer for CSV
                        answer_str = resp.extracted_answer
                        if not isinstance(answer_str, str):
                            answer_str = json.dumps(answer_str, default=str)

                        gt_str = resp.ground_truth
                        if not isinstance(gt_str, str):
                            gt_str = json.dumps(gt_str, default=str)

                        all_rows.append({
                            "trial": trial,
                            "task_id": resp.task_id,
                            "task_name": task.task_name,
                            "category": task.category,
                            "stimulus_id": resp.stimulus_id,
                            "condition": resp.condition,
                            "extracted_answer": answer_str[:500],
                            "ground_truth": gt_str[:500],
                            "score": resp.score,
                            "is_correct": resp.is_correct,
                            "gaming_hypothesis": task.gaming_hypothesis[:300],
                            "metadata_json": json.dumps(resp.metadata, default=str)[:2000],
                        })

                # Flush after each task completes all stimuli×conditions
                _flush_csv(csv_path, all_rows)

            completed_trials = trial

    except KeyboardInterrupt:
        logger.warning("Interrupted after %d completed trial(s)", completed_trials)
    except Exception:
        logger.warning("Error after %d completed trial(s)", completed_trials)
        raise
    finally:
        meta["n_trials_completed"] = completed_trials
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    logger.info("Raw outputs  → %s", raw_path)
    logger.info("Score table  → %s (%d trials)", csv_path, completed_trials)
    logger.info("Metadata     → %s", meta_path)

    return {"raw": raw_path, "csv": csv_path, "meta": meta_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run behavioral consciousness indicator tasks.",
    )
    parser.add_argument("--provider", default="openrouter",
                        choices=["anthropic", "openai", "openrouter"])
    parser.add_argument("--model", default="deepseek/deepseek-r1-0528:free")
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tasks", default="all",
                        help="Comma-separated task IDs or 'all' (default: all)")
    parser.add_argument("--stimuli-per-task", type=int, default=5,
                        help="Number of stimuli per task (default: 5)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.tasks == "all":
        task_ids = list(TASK_REGISTRY.keys())
    else:
        task_ids = [t.strip() for t in args.tasks.split(",")]

    cfg = ExperimentConfig(
        provider=args.provider,
        model=args.model,
        n_trials=args.n_trials,
        seed=args.seed,
        temperature=args.temperature,
    )

    if not cfg.api_key:
        key_var = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPEN_ROUTER_API_KEY",
        }.get(cfg.provider, "API_KEY")
        print(f"ERROR: {key_var} not set.", file=sys.stderr)
        sys.exit(1)

    outputs = run_behavioral(cfg, task_ids, args.stimuli_per_task)
    print("\nResults saved:")
    for label, path in outputs.items():
        print(f"  {label:5s} → {path}")


if __name__ == "__main__":
    main()
