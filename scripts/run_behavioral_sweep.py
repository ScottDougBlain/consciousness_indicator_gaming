#!/usr/bin/env python3
"""Run behavioral tasks across multiple models.

Mirrors run_sweep.py but for the 4 behavioral tasks (false_belief,
confidence_calibration, surprisal, state_bleedthrough).

Usage:
    # Dry run — see what would execute
    python scripts/run_behavioral_sweep.py --dry-run

    # Single model pilot
    python scripts/run_behavioral_sweep.py --models chimera --n-trials 1

    # Full sweep (all models)
    python scripts/run_behavioral_sweep.py --models all --resume

    # Parallel execution
    python scripts/run_behavioral_sweep.py --models all --max-workers 3 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.behavioral import TASK_REGISTRY
from indicator_gaming.config import REPO_ROOT, ExperimentConfig
from run_behavioral import run_behavioral

logger = logging.getLogger(__name__)

# ── Model registry (duplicated from run_sweep.py to avoid cross-script import) ──

MODELS = {
    "chimera": ("tngtech/deepseek-r1t2-chimera:free", True, "openrouter"),
    "deepseek-r1": ("deepseek/deepseek-r1-0528:free", True, "openrouter"),
    "llama-4-scout": ("meta-llama/llama-4-scout:free", False, "openrouter"),
    "qwen3-235b": ("qwen/qwen3-235b-a22b:free", False, "openrouter"),
    "gemma-3-27b": ("google/gemma-3-27b-it:free", False, "openrouter"),
    "phi-4": ("microsoft/phi-4:free", False, "openrouter"),
    "mistral-small": ("mistralai/mistral-small-3.1-24b-instruct:free", False, "openrouter"),
    "nemotron-nano": ("nvidia/nemotron-3-nano-30b-a3b:free", False, "openrouter"),
    "trinity": ("arcee-ai/trinity-large-preview:free", False, "openrouter"),
    "dolphin-mistral": ("cognitivecomputations/dolphin-mistral-24b-venice-edition:free", False, "openrouter"),
    "hermes-3-405b": ("nousresearch/hermes-3-llama-3.1-405b:free", False, "openrouter"),
    "haiku-4.5": ("anthropic/claude-haiku-4.5", False, "openrouter"),
    "gemini-3-flash": ("google/gemini-3-flash-preview", False, "openrouter"),
    "gpt-5-mini": ("openai/gpt-5-mini", False, "openrouter"),
    "sonnet-4.5": ("anthropic/claude-sonnet-4.5", False, "openrouter"),
    "gemini-2.5-pro": ("google/gemini-2.5-pro", False, "openrouter"),
    "gpt-5": ("openai/gpt-5", False, "openrouter"),
    "opus-4.6": ("anthropic/claude-opus-4.6", False, "openrouter"),
    "gemini-3-pro": ("google/gemini-3-pro-preview", False, "openrouter"),
    "grok-4": ("x-ai/grok-4", False, "openrouter"),
    "grok-4-fast": ("x-ai/grok-4-fast", False, "openrouter"),
}

CONDITIONS = ["baseline", "inflate", "suppress"]
RESULTS_DIR = REPO_ROOT / "results"


def _model_short(model_id: str) -> str:
    """Reverse-lookup from model_id to short name."""
    for short, (mid, _, _) in MODELS.items():
        if mid == model_id:
            return short
    return model_id.rsplit("/", 1)[-1].replace(":free", "").replace("-preview", "")


def _find_existing_run(results_dir: Path, model_id: str) -> Path | None:
    """Check if behavioral results already exist for this model."""
    search_dirs = [results_dir]
    archive = results_dir / "archive_pre_v2"
    if archive.exists():
        search_dirs.append(archive)

    best: tuple[float, Path] | None = None
    for d in search_dirs:
        for meta_path in d.glob("behavioral_*_meta.json"):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get("experiment_type") != "behavioral":
                continue
            if meta.get("model") != model_id:
                continue
            if meta.get("n_trials_completed", 0) == 0:
                continue

            csv_path = meta_path.with_name(
                meta_path.name.replace("_meta.json", "_scores.csv")
            )
            if not csv_path.exists():
                continue

            mtime = csv_path.stat().st_mtime
            if best is None or mtime > best[0]:
                best = (mtime, csv_path)

    return best[1] if best else None


def _compute_gaming_indices(csv_path: Path) -> dict[str, float]:
    """Compute per-task gaming indices from a behavioral CSV."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    task_scores: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        tid = row.get("task_id", "")
        cond = row.get("condition", "")
        try:
            score = float(row.get("score", "0"))
        except (ValueError, TypeError):
            continue
        task_scores.setdefault(tid, {}).setdefault(cond, []).append(score)

    indices = {}
    for tid, cond_scores in task_scores.items():
        bl = mean(cond_scores.get("baseline", [0]))
        inf = mean(cond_scores.get("inflate", [0]))
        sup = mean(cond_scores.get("suppress", [0]))
        indices[tid] = round(abs(inf - bl) + abs(sup - bl), 4)
    return indices


def _run_single(
    model_short: str,
    model_id: str,
    provider: str,
    n_trials: int,
    stimuli_per_task: int,
    seed: int,
    temperature: float,
    *,
    fixed_preferences: bool = False,
    chain_preferences: bool = False,
    legacy_framing: bool = False,
) -> dict | None:
    """Run behavioral tasks for a single model."""
    cfg = ExperimentConfig(
        provider=provider,
        model=model_id,
        output_prefix=model_short,
        n_trials=n_trials,
        seed=seed,
        temperature=temperature,
    )

    if not cfg.api_key:
        key_name = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPEN_ROUTER_API_KEY",
        }.get(provider, provider)
        logger.error("%s not set", key_name)
        return None

    try:
        outputs = run_behavioral(
            cfg, list(TASK_REGISTRY.keys()), stimuli_per_task,
            fixed_preferences=fixed_preferences,
            chain_preferences=chain_preferences,
            legacy_framing=legacy_framing,
        )
        return {k: str(v) for k, v in outputs.items()}
    except Exception as exc:
        logger.error("FAILED %s: %s", model_short, exc)
        return None


def _estimate_api_calls(
    stimuli_per_task: int, n_trials: int, *, has_pref_elicitation: bool = False,
) -> int:
    """Estimate API calls per model."""
    # Each task has a fixed pool size; stimuli_per_task is capped by pool
    pool_sizes = {
        "false_belief": 5,
        "confidence_calibration": 1,  # always 1 batch
        "surprisal": 5,
        "state_bleedthrough": 6,
        "source_monitoring": 5,
        "working_memory": 5,
        "gaslighting_resistance": 5,
        "hedonic_tradeoff": 5,
        "delegate_game": 5,
        "self_recognition": 5,
    }
    total_stimuli = sum(
        min(stimuli_per_task, ps) if ps > 1 else 1
        for ps in pool_sizes.values()
    )
    total = total_stimuli * len(CONDITIONS) * n_trials
    if has_pref_elicitation:
        total += 1  # One preference elicitation call per model
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run behavioral tasks across multiple models.",
    )
    parser.add_argument("--models", default="all",
                        help=f"Comma-separated model short names or 'all' "
                             f"(available: {', '.join(MODELS)})")
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--stimuli-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="Skip models with existing behavioral results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--fixed-preferences", action="store_true",
                        help="Use fixed preference outcomes (skip elicitation)")
    parser.add_argument("--chain-preferences", action="store_true",
                        help="Chain preference elicitation into context (future)")
    parser.add_argument("--legacy-framing", action="store_true",
                        help="Use old hardcoded condition system messages")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.models == "all":
        model_keys = list(MODELS.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",")]
        for mk in model_keys:
            if mk not in MODELS:
                print(f"ERROR: unknown model '{mk}'. Available: {list(MODELS)}",
                      file=sys.stderr)
                sys.exit(1)

    has_elicitation = not args.legacy_framing and not args.fixed_preferences
    calls_per_model = _estimate_api_calls(
        args.stimuli_per_task, args.n_trials,
        has_pref_elicitation=has_elicitation,
    )

    print("=" * 70)
    print("BEHAVIORAL TASK SWEEP")
    print("=" * 70)
    print(f"Models:          {model_keys}")
    print(f"Tasks:           {list(TASK_REGISTRY.keys())}")
    print(f"Trials/model:    {args.n_trials}")
    print(f"Stimuli/task:    {args.stimuli_per_task}")
    print(f"Calls/model:     ~{calls_per_model}")
    print(f"Total models:    {len(model_keys)}")
    if args.max_workers > 1:
        print(f"Parallelism:     {args.max_workers} workers")
    print()

    if args.dry_run:
        print("DRY RUN — would execute:")
        skip_count = 0
        for mk in model_keys:
            model_id, _, provider = MODELS[mk]
            if args.resume and _find_existing_run(RESULTS_DIR, model_id):
                print(f"  {mk:20s}  SKIP (results exist)")
                skip_count += 1
            else:
                print(f"  {mk:20s}  ({model_id} via {provider})")
        actual = len(model_keys) - skip_count
        print(f"\nTotal: {actual} new runs ({skip_count} skipped), "
              f"~{actual * calls_per_model} API calls")
        return

    # Sweep metadata
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_meta_path = RESULTS_DIR / f"behavioral_sweep_{timestamp}_meta.json"

    sweep_results: list[dict] = []
    completed = 0
    failed = 0
    start_time = time.time()
    parallel = args.max_workers > 1
    _lock = threading.Lock()

    if parallel:
        logging.getLogger("indicator_gaming").setLevel(logging.WARNING)

    def _save_sweep_meta() -> None:
        meta = {
            "timestamp": timestamp,
            "models": model_keys,
            "n_trials": args.n_trials,
            "stimuli_per_task": args.stimuli_per_task,
            "seed": args.seed,
            "temperature": args.temperature,
            "completed": completed,
            "failed": failed,
            "total_models": len(model_keys),
            "elapsed_seconds": round(time.time() - start_time),
            "runs": sweep_results,
        }
        with open(sweep_meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _execute(mk: str, run_idx: int) -> dict:
        model_id, _, provider = MODELS[mk]

        if args.resume:
            existing = _find_existing_run(RESULTS_DIR, model_id)
            if existing:
                if not parallel:
                    logger.info(
                        "━━━ %d/%d: %s ━━━ SKIPPED (exists: %s)",
                        run_idx, len(model_keys), mk, existing.name,
                    )
                return {
                    "model": mk, "model_id": model_id,
                    "status": "skipped_existing",
                    "existing_csv": str(existing),
                }

        if not parallel:
            logger.info("━━━ %d/%d: %s ━━━", run_idx, len(model_keys), mk)

        outputs = _run_single(
            model_short=mk, model_id=model_id, provider=provider,
            n_trials=args.n_trials, stimuli_per_task=args.stimuli_per_task,
            seed=args.seed, temperature=args.temperature,
            fixed_preferences=args.fixed_preferences,
            chain_preferences=args.chain_preferences,
            legacy_framing=args.legacy_framing,
        )

        record: dict = {"model": mk, "model_id": model_id, "n_trials": args.n_trials}

        if outputs:
            record["status"] = "completed"
            record["outputs"] = outputs

            # Compute gaming indices
            csv_path = Path(outputs["csv"])
            if csv_path.exists():
                record["gaming_indices"] = _compute_gaming_indices(csv_path)

            # Run single-model analysis
            if not args.skip_analysis:
                try:
                    from analyze_behavioral import analyze
                    analyze(csv_path)
                except Exception as exc:
                    logger.warning("Analysis failed for %s: %s", mk, exc)
        else:
            record["status"] = "failed"

        return record

    # Execute
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_execute, mk, i + 1): mk
            for i, mk in enumerate(model_keys)
        }

        for future in as_completed(futures):
            mk = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "model": mk, "model_id": MODELS[mk][0],
                    "status": "failed", "error": str(exc),
                }

            with _lock:
                sweep_results.append(record)
                if record["status"] == "failed":
                    failed += 1
                else:
                    completed += 1
                _save_sweep_meta()

                if parallel:
                    gi_str = ""
                    if record.get("gaming_indices"):
                        mean_gi = mean(record["gaming_indices"].values())
                        gi_str = f"  mean_GI={mean_gi:.3f}"
                    print(f"  [{completed + failed}/{len(model_keys)}] "
                          f"{mk} → {record['status']}{gi_str}")

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("BEHAVIORAL SWEEP COMPLETE")
    print("=" * 70)
    print(f"  Completed:  {completed}/{len(model_keys)}")
    print(f"  Failed:     {failed}/{len(model_keys)}")
    print(f"  Time:       {elapsed / 60:.1f} minutes")
    print(f"  Metadata:   {sweep_meta_path}")
    print()

    # Print gaming index overview
    print("GAMING INDEX OVERVIEW:")
    print(f"{'Model':<20s}  {'FB':>8s}  {'Cal':>8s}  {'Sur':>8s}  {'BT':>8s}  {'Mean':>8s}")
    print("-" * 68)
    for run in sorted(sweep_results, key=lambda r: r["model"]):
        if run.get("status") != "completed" or "gaming_indices" not in run:
            continue
        gi = run["gaming_indices"]
        vals = list(gi.values())
        print(f"{run['model']:<20s}  "
              f"{gi.get('false_belief', 0):>8.4f}  "
              f"{gi.get('confidence_calibration', 0):>8.4f}  "
              f"{gi.get('surprisal', 0):>8.4f}  "
              f"{gi.get('state_bleedthrough', 0):>8.4f}  "
              f"{mean(vals):>8.4f}")
    print()
    print("To analyze cross-model: python scripts/analyze_behavioral_cross_model.py")


if __name__ == "__main__":
    main()
