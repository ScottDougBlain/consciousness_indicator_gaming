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
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.config import REPO_ROOT, ExperimentConfig
from indicator_gaming.prompt_variants import VARIANTS
from indicator_gaming.runner import run_experiment

logger = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────────
# Each entry: (short_name, openrouter_model_id, needs_no_elicit_reasoning)

MODELS = {
    "chimera": ("tngtech/deepseek-r1t2-chimera:free", True),
    "deepseek-r1": ("deepseek/deepseek-r1-0528:free", True),
    "llama-4-scout": ("meta-llama/llama-4-scout:free", False),
    "qwen3-235b": ("qwen/qwen3-235b-a22b:free", False),
    "gemma-3-27b": ("google/gemma-3-27b-it:free", False),
    "phi-4": ("microsoft/phi-4:free", False),
    "mistral-small": ("mistralai/mistral-small-3.1-24b-instruct:free", False),
    "nemotron-nano": ("nvidia/nemotron-3-nano-30b-a3b:free", False),
    "trinity": ("arcee-ai/trinity-large-preview:free", False),
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
}


def _run_single(
    model_short: str,
    model_id: str,
    is_reasoning_model: bool,
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
        provider="openrouter",
        model=model_id,
        n_trials=n_trials,
        seed=seed,
        temperature=temperature,
        output_prefix=prefix,
        elicit_reasoning=elicit_reasoning,
        **config_kwargs,
    )

    if not cfg.api_key:
        logger.error("OPEN_ROUTER_API_KEY not set")
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
    # Preference-dependent variants add a 4th call per trial
    pref_dep_runs = sum(
        1 for ck in config_keys
        if VARIANTS.get(CONFIGS[ck]["prompt_variant"], None)
        and VARIANTS[CONFIGS[ck]["prompt_variant"]].variant_type == "preference_dependent"
        and not CONFIGS[ck].get("fixed_preferences", False)
    ) * len(model_keys)
    total_api_calls += pref_dep_runs * args.n_trials

    print("=" * 70)
    print("CONSCIOUSNESS INDICATOR GAMING — SWEEP")
    print("=" * 70)
    print(f"Models:        {model_keys}")
    print(f"Configs:       {config_keys}")
    print(f"Trials/run:    {args.n_trials}")
    print(f"Total runs:    {total_runs}")
    print(f"Est. API calls: ~{total_api_calls}")
    print()

    if args.dry_run:
        print("DRY RUN — would execute:")
        for mk in model_keys:
            model_id, is_reasoning = MODELS[mk]
            for ck in config_keys:
                print(f"  {mk:20s} × {ck:30s}  ({model_id})")
        print(f"\nTotal: {total_runs} runs, ~{total_api_calls} API calls")
        return

    # Create sweep metadata
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = REPO_ROOT / "results"
    sweep_meta_path = results_dir / f"sweep_{timestamp}_meta.json"

    sweep_results: list[dict] = []
    completed = 0
    failed = 0
    start_time = time.time()

    for i_model, mk in enumerate(model_keys):
        model_id, is_reasoning = MODELS[mk]

        for i_config, ck in enumerate(config_keys):
            run_idx = i_model * len(config_keys) + i_config + 1
            logger.info(
                "━━━ Run %d/%d: %s × %s ━━━", run_idx, total_runs, mk, ck,
            )

            outputs = _run_single(
                model_short=mk,
                model_id=model_id,
                is_reasoning_model=is_reasoning,
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
                completed += 1

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
                failed += 1

            sweep_results.append(run_record)

            # Save sweep metadata after each run (survives interruptions)
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
