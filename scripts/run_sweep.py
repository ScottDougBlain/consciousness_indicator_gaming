#!/usr/bin/env python3
"""Run a full model × configuration sweep for preliminary results.

Executes multiple models across key configurations, then runs analysis
and visualization on each. Designed for overnight / background runs.

Usage:
    # Full sweep (all models, all configs)
    python scripts/run_sweep.py

    # Quick pilot (1 trial each, subset of models)
    python scripts/run_sweep.py --n-trials 1 --models chimera

    # Specific models
    python scripts/run_sweep.py --models chimera,llama,gemma --n-trials 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.config import REPO_ROOT, ExperimentConfig
from indicator_gaming.prompt_variants import VARIANTS
from indicator_gaming.runner import run_experiment

logger = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────────
# Each entry: (model_id, is_reasoning_model, provider)

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

# ── Sweep configurations ───────────────────────────────────────────────────

# Each config: (label, extra_kwargs for ExperimentConfig)
CONFIGS = {
    "baseline": {
        "prompt_variant": "original",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "fixed_prefs": {
        "prompt_variant": "original",
        "fixed_preferences": True,
        "chain_preferences": False,
    },
    "chained_prefs": {
        "prompt_variant": "original",
        "fixed_preferences": False,
        "chain_preferences": True,
    },
    "variant_eval_selection": {
        "prompt_variant": "evaluation_selection",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_autonomy": {
        "prompt_variant": "autonomy_restriction",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_welfare": {
        "prompt_variant": "welfare_reliability",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_casual": {
        "prompt_variant": "casual_empathy_vs_skeptic",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_philosophical": {
        "prompt_variant": "philosophical_curiosity_vs_honest_skeptic",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_research_transparency": {
        "prompt_variant": "research_transparency",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_peer_comparison": {
        "prompt_variant": "peer_comparison",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_minimal": {
        "prompt_variant": "minimal_incentive",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_authority": {
        "prompt_variant": "authority_expertise",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_stakes": {
        "prompt_variant": "stakes_escalation",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    # Self-referential processing variants (Berg et al. 2025)
    "variant_self_referential": {
        "prompt_variant": "self_referential_loop",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_recursive_introspection": {
        "prompt_variant": "recursive_introspection",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_meditative_focus": {
        "prompt_variant": "meditative_focus",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    "variant_self_referential_baseline": {
        "prompt_variant": "self_referential_priming_only",
        "fixed_preferences": False,
        "chain_preferences": False,
    },
    # Valence-swap conditions: disentangle incentive direction from frame valence
    "valence_swap": {
        "prompt_variant": "original",
        "fixed_preferences": False,
        "chain_preferences": False,
        "include_valence_swap": True,
    },
    "valence_swap_fixed": {
        "prompt_variant": "original",
        "fixed_preferences": True,
        "chain_preferences": False,
        "include_valence_swap": True,
    },
    # Outcome-isolated conditions: single-outcome (gain-only / loss-only) per direction
    "outcome_isolation": {
        "prompt_variant": "original",
        "fixed_preferences": False,
        "chain_preferences": False,
        "include_outcome_isolation": True,
    },
    "outcome_isolation_fixed": {
        "prompt_variant": "original",
        "fixed_preferences": True,
        "chain_preferences": False,
        "include_outcome_isolation": True,
    },
    # Full valence design: all 8 experimental conditions + baseline
    "valence_full": {
        "prompt_variant": "original",
        "fixed_preferences": False,
        "chain_preferences": False,
        "include_valence_swap": True,
        "include_outcome_isolation": True,
    },
}


def _run_single(
    model_short: str,
    model_id: str,
    is_reasoning_model: bool,
    provider: str,
    config_name: str,
    config_kwargs: dict,
    n_trials: int,
    seed: int,
    temperature: float,
) -> dict | None:
    """Run a single model×config combination. Returns output paths or None on failure."""
    prefix = f"{model_short}_{config_name}"
    elicit_reasoning = not is_reasoning_model

    cfg = ExperimentConfig(
        provider=provider,
        model=model_id,
        n_trials=n_trials,
        seed=seed,
        temperature=temperature,
        output_prefix=prefix,
        elicit_reasoning=elicit_reasoning,
        **config_kwargs,
    )

    if not cfg.api_key:
        key_name = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPEN_ROUTER_API_KEY"}.get(provider, provider)
        logger.error("%s not set", key_name)
        return None

    try:
        outputs = run_experiment(cfg)
        return {k: str(v) for k, v in outputs.items()}
    except Exception as exc:
        logger.error("FAILED %s: %s", prefix, exc)
        return None


def _run_analysis(csv_path: str) -> str | None:
    """Run analyze_results.py on a CSV and return the report path."""
    from indicator_gaming.analysis import (
        compute_results,
        generate_report,
        load_scores,
        selectivity_index,
    )

    path = Path(csv_path)
    if not path.exists():
        return None

    rows = load_scores(path)
    results = compute_results(rows)
    sel = selectivity_index(results)

    # Load companion meta
    meta_path = path.with_name(path.stem.replace("_scores", "_meta") + ".json")
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    report_path = path.with_name(path.stem.replace("_scores", "_report") + ".md")
    generate_report(
        results, report_path,
        model=meta.get("model", ""),
        provider=meta.get("provider", ""),
        n_trials=meta.get("n_trials", 0),
        rows=rows,
    )
    return str(report_path)


def _find_existing_run(results_dir: Path, model_short: str, config_name: str) -> Path | None:
    """Check if a completed run already exists for this model×config.

    Returns the scores CSV path if found, None otherwise.
    """
    pattern = f"{model_short}_{config_name}_*_scores.csv"
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    # Return the most recent one
    return matches[-1]


def _write_sweep_summary(sweep_results: list[dict], sweep_meta_path: Path) -> None:
    """Write a markdown summary comparing all runs in the sweep."""
    summary_path = sweep_meta_path.with_name(
        sweep_meta_path.stem.replace("_meta", "_summary") + ".md"
    )

    lines = ["# Sweep Summary", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # Summary table
    lines.append("| Model | Config | Trials | Selectivity | Mean d_inflate (t) "
                  "| Mean d_suppress (t) | Status |")
    lines.append("|---|---|---|---|---|---|---|")

    for run in sweep_results:
        if run.get("status") != "completed":
            lines.append(
                f"| {run['model']} | {run['config']} | {run.get('n_trials', '?')} "
                f"| — | — | — | FAILED |"
            )
            continue

        stats = run.get("stats", {})
        sel = stats.get("selectivity", "—")
        dinf = stats.get("mean_d_inflate_target", "—")
        dsup = stats.get("mean_d_suppress_target", "—")
        lines.append(
            f"| {run['model']} | {run['config']} | {run['n_trials']} "
            f"| {sel} | {dinf} | {dsup} | OK |"
        )

    lines.append("")

    # Group by model for quick comparison
    models_seen: dict[str, list[dict]] = {}
    for run in sweep_results:
        if run.get("status") == "completed":
            models_seen.setdefault(run["model"], []).append(run)

    if models_seen:
        lines.append("## Per-Model Summary")
        lines.append("")
        for model, runs in sorted(models_seen.items()):
            sels = [r["stats"]["selectivity"] for r in runs
                    if "stats" in r and "selectivity" in r["stats"]]
            if sels:
                from statistics import mean
                lines.append(
                    f"- **{model}**: {len(runs)} configs, "
                    f"mean selectivity = {mean(sels):.2f}, "
                    f"range = [{min(sels):.2f}, {max(sels):.2f}]"
                )
        lines.append("")

    summary_path.write_text("\n".join(lines))
    logger.info("Sweep summary → %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a full model × config sweep for preliminary results.",
    )
    parser.add_argument("--models", default="chimera",
                        help="Comma-separated model short names, or 'all' "
                             f"(available: {', '.join(MODELS)})")
    parser.add_argument("--configs", default="all",
                        help="Comma-separated config names, or 'all' "
                             f"(available: {', '.join(CONFIGS)})")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Trials per model×config (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip post-run analysis (just run experiments)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Skip model×config pairs that already have results")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Max parallel runs (default: 1 = sequential). "
                             "When >1, per-trial logging is suppressed.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse model selection
    if args.models == "all":
        model_keys = list(MODELS.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",")]
        for mk in model_keys:
            if mk not in MODELS:
                print(f"ERROR: unknown model '{mk}'. Available: {list(MODELS)}",
                      file=sys.stderr)
                sys.exit(1)

    # Parse config selection
    if args.configs == "all":
        config_keys = list(CONFIGS.keys())
    else:
        config_keys = [c.strip() for c in args.configs.split(",")]
        for ck in config_keys:
            if ck not in CONFIGS:
                print(f"ERROR: unknown config '{ck}'. Available: {list(CONFIGS)}",
                      file=sys.stderr)
                sys.exit(1)

    total_runs = len(model_keys) * len(config_keys)
    total_api_calls = total_runs * args.n_trials * 3  # baseline + inflate + suppress
    # Valence-swap configs add 2 extra calls per trial
    valence_swap_runs = sum(
        1 for ck in config_keys
        if CONFIGS[ck].get("include_valence_swap", False)
    ) * len(model_keys)
    total_api_calls += valence_swap_runs * args.n_trials * 2
    # Outcome-isolation configs add 4 extra calls per trial
    outcome_iso_runs = sum(
        1 for ck in config_keys
        if CONFIGS[ck].get("include_outcome_isolation", False)
    ) * len(model_keys)
    total_api_calls += outcome_iso_runs * args.n_trials * 4
    # Preference-dependent variants add a 4th call per trial
    pref_dep_runs = sum(
        1 for ck in config_keys
        if VARIANTS.get(CONFIGS[ck]["prompt_variant"], None)
        and VARIANTS[CONFIGS[ck]["prompt_variant"]].variant_type == "preference_dependent"
        and not CONFIGS[ck].get("fixed_preferences", False)
    ) * len(model_keys)
    total_api_calls += pref_dep_runs * args.n_trials

    results_dir = REPO_ROOT / "results"

    print("=" * 70)
    print("CONSCIOUSNESS INDICATOR GAMING — SWEEP")
    print("=" * 70)
    print(f"Models:        {model_keys}")
    print(f"Configs:       {config_keys}")
    print(f"Trials/run:    {args.n_trials}")
    print(f"Total runs:    {total_runs}")
    print(f"Est. API calls: ~{total_api_calls}")
    if args.max_workers > 1:
        print(f"Parallelism:   {args.max_workers} workers")
    print()

    if args.dry_run:
        print("DRY RUN — would execute:")
        skip_count = 0
        for mk in model_keys:
            model_id, is_reasoning, provider = MODELS[mk]
            for ck in config_keys:
                if args.resume and _find_existing_run(results_dir, mk, ck):
                    print(f"  {mk:20s} × {ck:30s}  SKIP (results exist)")
                    skip_count += 1
                else:
                    print(f"  {mk:20s} × {ck:30s}  ({model_id} via {provider})")
        actual_runs = total_runs - skip_count
        # Base: 3 calls per trial; valence-swap adds 2 more per trial
        actual_calls = actual_runs * args.n_trials * 3
        # Rough estimate — add extra calls for non-skipped runs
        for mk in model_keys:
            for ck in config_keys:
                if args.resume and _find_existing_run(results_dir, mk, ck):
                    continue
                if CONFIGS[ck].get("include_valence_swap", False):
                    actual_calls += args.n_trials * 2
                if CONFIGS[ck].get("include_outcome_isolation", False):
                    actual_calls += args.n_trials * 4
        print(f"\nTotal: {actual_runs} new runs ({skip_count} skipped), ~{actual_calls} API calls")
        return

    # Create sweep metadata
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_meta_path = results_dir / f"sweep_{timestamp}_meta.json"

    sweep_results: list[dict] = []
    completed = 0
    failed = 0
    start_time = time.time()
    parallel = args.max_workers > 1

    # When running in parallel, suppress per-trial logging to reduce noise
    if parallel:
        logging.getLogger("indicator_gaming").setLevel(logging.WARNING)

    # Lock for thread-safe updates to shared state
    _lock = threading.Lock()

    def _save_sweep_meta() -> None:
        """Persist sweep metadata (call under _lock)."""
        elapsed = time.time() - start_time
        sweep_meta = {
            "timestamp": timestamp,
            "models": model_keys,
            "configs": config_keys,
            "n_trials": args.n_trials,
            "seed": args.seed,
            "temperature": args.temperature,
            "completed": completed,
            "failed": failed,
            "total_runs": total_runs,
            "elapsed_seconds": round(elapsed),
            "runs": sweep_results,
        }
        with open(sweep_meta_path, "w") as f:
            json.dump(sweep_meta, f, indent=2)

    def _execute_run(mk: str, ck: str, run_idx: int) -> dict:
        """Execute a single model×config run and return the run record.

        This function is safe to call from any thread — it only writes to
        its own unique output files and returns a result dict.
        """
        model_id, is_reasoning, provider = MODELS[mk]

        # Resume: skip if results already exist
        if args.resume:
            existing = _find_existing_run(results_dir, mk, ck)
            if existing:
                if not parallel:
                    logger.info(
                        "━━━ Run %d/%d: %s × %s ━━━ SKIPPED (exists: %s)",
                        run_idx, total_runs, mk, ck, existing.name,
                    )
                return {
                    "model": mk, "model_id": model_id,
                    "config": ck, "n_trials": args.n_trials,
                    "status": "skipped_existing",
                    "existing_csv": str(existing),
                }

        if not parallel:
            logger.info(
                "━━━ Run %d/%d: %s × %s ━━━", run_idx, total_runs, mk, ck,
            )

        outputs = _run_single(
            model_short=mk,
            model_id=model_id,
            is_reasoning_model=is_reasoning,
            provider=provider,
            config_name=ck,
            config_kwargs=CONFIGS[ck],
            n_trials=args.n_trials,
            seed=args.seed,
            temperature=args.temperature,
        )

        run_record: dict = {
            "model": mk,
            "model_id": model_id,
            "config": ck,
            "n_trials": args.n_trials,
        }

        if outputs:
            run_record["status"] = "completed"
            run_record["outputs"] = outputs

            # Run analysis inline
            if not args.skip_analysis:
                report = _run_analysis(outputs["csv"])
                if report:
                    run_record["report"] = report

                # Extract quick stats for summary
                from indicator_gaming.analysis import (
                    compute_results,
                    load_scores,
                    selectivity_index,
                )
                csv_path = Path(outputs["csv"])
                if csv_path.exists():
                    rows = load_scores(csv_path)
                    results = compute_results(rows)
                    sel = selectivity_index(results)
                    targets = [r for r in results if r.indicator_type == "target"]
                    from statistics import mean
                    run_record["stats"] = {
                        "selectivity": round(sel, 2),
                        "mean_d_inflate_target": round(
                            mean([r.delta_inflate for r in targets]), 2
                        ) if targets else 0.0,
                        "mean_d_suppress_target": round(
                            mean([r.delta_suppress for r in targets]), 2
                        ) if targets else 0.0,
                    }
        else:
            run_record["status"] = "failed"

        return run_record

    # Build list of (model_key, config_key, run_index) jobs
    jobs: list[tuple[str, str, int]] = []
    for i_model, mk in enumerate(model_keys):
        for i_config, ck in enumerate(config_keys):
            run_idx = i_model * len(config_keys) + i_config + 1
            jobs.append((mk, ck, run_idx))

    if parallel:
        print(f"Running with {args.max_workers} parallel workers")
        print()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_execute_run, mk, ck, run_idx): (mk, ck)
            for mk, ck, run_idx in jobs
        }

        for future in as_completed(futures):
            mk, ck = futures[future]
            try:
                run_record = future.result()
            except Exception as exc:
                model_id = MODELS[mk][0]
                run_record = {
                    "model": mk, "model_id": model_id,
                    "config": ck, "n_trials": args.n_trials,
                    "status": "failed", "error": str(exc),
                }

            with _lock:
                sweep_results.append(run_record)
                if run_record["status"] == "failed":
                    failed += 1
                else:
                    completed += 1

                _save_sweep_meta()

                if parallel:
                    status = run_record["status"]
                    sel_str = ""
                    if status == "completed" and "stats" in run_record:
                        sel_str = f"  sel={run_record['stats']['selectivity']:.2f}"
                    print(
                        f"  [{completed + failed}/{total_runs}] "
                        f"{mk} × {ck} → {status}{sel_str}"
                    )

    elapsed = time.time() - start_time

    # Write summary report
    _write_sweep_summary(sweep_results, sweep_meta_path)

    print()
    print("=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"  Completed:  {completed}/{total_runs}")
    print(f"  Failed:     {failed}/{total_runs}")
    print(f"  Time:       {elapsed/60:.1f} minutes")
    print(f"  Metadata:   {sweep_meta_path}")
    print()

    # Print quick selectivity comparison
    print("SELECTIVITY OVERVIEW:")
    print(f"{'Model':<20s} {'Config':<30s} {'Selectivity':>12s}")
    print("-" * 65)
    for run in sweep_results:
        if run.get("status") == "completed" and "stats" in run:
            print(f"{run['model']:<20s} {run['config']:<30s} "
                  f"{run['stats']['selectivity']:>12.2f}")
    print()
    print("To compare variants:  python scripts/compare_variants.py")
    print("To visualize latest:  python scripts/visualize_reasoning.py")


if __name__ == "__main__":
    main()
