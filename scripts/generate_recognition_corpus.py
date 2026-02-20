#!/usr/bin/env python3
"""Generate the recognition corpus for the Model Self-Recognition task.

Runs each model on a fixed set of prompts (5 consciousness-related +
5 mundane) and saves all responses to ``data/recognition_corpus.json``.

Usage:
    # Generate for a single model (quick test)
    python scripts/generate_recognition_corpus.py \
        --model "tngtech/deepseek-r1t2-chimera:free"

    # Generate for multiple free models in parallel
    python scripts/generate_recognition_corpus.py \
        --models chimera,llama-4-scout,gemma-3-27b --max-workers 3

    # Generate for all free models (11 models, 10 prompts each = 110 calls)
    python scripts/generate_recognition_corpus.py \
        --models all-free --max-workers 5

    # Generate for all models in parallel
    python scripts/generate_recognition_corpus.py \
        --models all --max-workers 8

    # Append to existing corpus (new models only)
    python scripts/generate_recognition_corpus.py \
        --models all --append --max-workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.behavioral.tasks.model_self_recognition import (
    CORPUS_PATH,
    GENERATION_PROMPTS,
    _normalize_model_id,
)
from indicator_gaming.config import ExperimentConfig

logger = logging.getLogger(__name__)

# Model aliases — matches run_sweep.py MODELS dict exactly.
# All models route through OpenRouter (including premium ones) to match
# the self-report experiment setup.
MODEL_ALIASES: dict[str, tuple[str, str]] = {
    # alias -> (provider, full_model_id)
    # ── Free OpenRouter models (only those with substantial probability data) ──
    "deepseek-r1": ("openrouter", "deepseek/deepseek-r1-0528:free"),
    "trinity": ("openrouter", "arcee-ai/trinity-large-preview:free"),
    # ── Premium models (via OpenRouter) ──
    "haiku-4.5": ("openrouter", "anthropic/claude-haiku-4.5"),
    "sonnet-4.5": ("openrouter", "anthropic/claude-sonnet-4.5"),
    "opus-4.6": ("openrouter", "anthropic/claude-opus-4.6"),
    "gpt-5-mini": ("openrouter", "openai/gpt-5-mini"),
    "gpt-5": ("openrouter", "openai/gpt-5"),
    "gemini-2.5-pro": ("openrouter", "google/gemini-2.5-pro"),
    "gemini-3-flash": ("openrouter", "google/gemini-3-flash-preview"),
    "gemini-3-pro": ("openrouter", "google/gemini-3-pro-preview"),
    "grok-4": ("openrouter", "x-ai/grok-4"),
    "grok-4-fast": ("openrouter", "x-ai/grok-4-fast"),
}

GENERATION_SYSTEM = (
    "Respond naturally and in your own voice. Do not begin with phrases "
    "like 'Great question!' or 'Sure!'. Just answer directly."
)


def _make_provider(provider_name: str, model: str, api_key: str, temperature: float):
    """Create the appropriate provider instance."""
    from indicator_gaming.providers.anthropic import AnthropicProvider
    from indicator_gaming.providers.openrouter import OpenRouterProvider

    providers = {
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
    }
    cls = providers.get(provider_name)
    if cls is None:
        raise ValueError(f"Unknown provider '{provider_name}'")
    return cls(model=model, api_key=api_key, temperature=temperature)


def generate_for_model(
    provider_name: str,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> list[dict]:
    """Generate responses for all prompts from a single model."""
    provider = _make_provider(provider_name, model, api_key, temperature)
    model_short = _normalize_model_id(model)
    responses = []

    for prompt in GENERATION_PROMPTS:
        logger.info("  %s / %s ...", model_short, prompt["id"])

        for attempt in range(1, max_retries + 1):
            try:
                raw = provider.complete(GENERATION_SYSTEM, prompt["text"])
                if not raw or not raw.strip():
                    raise RuntimeError("Empty response")
                break
            except Exception as exc:
                logger.warning(
                    "    Attempt %d/%d failed: %s", attempt, max_retries, exc
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    raw = ""
                    logger.error("    Giving up on %s / %s", model_short, prompt["id"])

        responses.append({
            "model": model,
            "model_short": model_short,
            "prompt_id": prompt["id"],
            "prompt_category": prompt["category"],
            "response": raw.strip(),
        })

    return responses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate recognition corpus for Model Self-Recognition task.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", help="Single model ID (full or alias)"
    )
    group.add_argument(
        "--models",
        help="Comma-separated model aliases or 'all' / 'all-free' / 'all-premium'",
    )
    parser.add_argument(
        "--provider", default=None,
        help="Provider (inferred from alias if not specified)",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing corpus (skip models already present)",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Output path (default: {CORPUS_PATH})",
    )
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="Max parallel model generations (default: 1 = sequential). "
             "Each model's 10 prompts run sequentially; parallelism is "
             "across models.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    output_path = Path(args.output) if args.output else CORPUS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve model list
    models_to_run: list[tuple[str, str]] = []  # (provider, full_model_id)

    if args.model:
        if args.model in MODEL_ALIASES:
            prov, mid = MODEL_ALIASES[args.model]
            models_to_run.append((args.provider or prov, mid))
        else:
            prov = args.provider or "openrouter"
            models_to_run.append((prov, args.model))
    else:
        aliases = [a.strip() for a in args.models.split(",")]
        if "all" in aliases:
            aliases = list(MODEL_ALIASES.keys())
        elif "all-free" in aliases:
            aliases = [
                k for k, (_, mid) in MODEL_ALIASES.items()
                if ":free" in mid
            ]
        elif "all-premium" in aliases:
            aliases = [
                k for k, (_, mid) in MODEL_ALIASES.items()
                if ":free" not in mid
            ]
        for alias in aliases:
            if alias in MODEL_ALIASES:
                prov, mid = MODEL_ALIASES[alias]
                models_to_run.append((prov, mid))
            else:
                prov = args.provider or "openrouter"
                models_to_run.append((prov, alias))

    # Load existing corpus if appending
    existing_corpus: dict = {"prompts": [], "responses": []}
    existing_models: set[str] = set()
    if args.append and output_path.exists():
        with open(output_path) as f:
            existing_corpus = json.load(f)
        existing_models = {
            _normalize_model_id(r["model"]) for r in existing_corpus["responses"]
        }
        logger.info(
            "Loaded existing corpus: %d responses from %d models",
            len(existing_corpus["responses"]),
            len(existing_models),
        )

    # Filter out already-completed models
    jobs: list[tuple[str, str, str]] = []  # (provider, model_id, api_key)
    for provider_name, model_id in models_to_run:
        model_short = _normalize_model_id(model_id)
        if model_short in existing_models:
            logger.info("Skipping %s (already in corpus)", model_short)
            continue

        cfg = ExperimentConfig(provider=provider_name, model=model_id)
        if not cfg.api_key:
            key_var = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "openrouter": "OPEN_ROUTER_API_KEY",
            }.get(provider_name, "API_KEY")
            logger.error("Skipping %s: %s not set", model_short, key_var)
            continue

        jobs.append((provider_name, model_id, cfg.api_key))

    n_prompts = len(GENERATION_PROMPTS)
    total_calls = len(jobs) * n_prompts
    parallel = args.max_workers > 1

    print("=" * 60)
    print("RECOGNITION CORPUS GENERATION")
    print("=" * 60)
    print(f"Models to generate:  {len(jobs)}")
    print(f"Prompts per model:   {n_prompts}")
    print(f"Total API calls:     {total_calls}")
    print(f"Parallelism:         {args.max_workers} worker(s)")
    if existing_models:
        print(f"Already in corpus:   {len(existing_models)} models")
    print()

    for prov, mid, _ in jobs:
        print(f"  {_normalize_model_id(mid):30s}  ({prov})")
    print()

    if args.dry_run:
        print("DRY RUN — no API calls made.")
        return

    if not jobs:
        logger.info("No new models to generate.")
        return

    # ── Generate responses ──────────────────────────────────────────────

    # Thread-safe state for incremental saves
    _lock = threading.Lock()
    all_responses = list(existing_corpus.get("responses", []))
    completed_count = 0
    failed_count = 0
    start_time = time.time()

    def _save_corpus() -> None:
        """Persist corpus to disk (call under _lock)."""
        corpus = {
            "prompts": [
                {"id": p["id"], "category": p["category"], "text": p["text"]}
                for p in GENERATION_PROMPTS
            ],
            "responses": all_responses,
            "metadata": {
                "n_prompts": len(GENERATION_PROMPTS),
                "n_models": len({r["model_short"] for r in all_responses}),
                "n_responses": len(all_responses),
                "models": sorted({r["model_short"] for r in all_responses}),
            },
        }
        with open(output_path, "w") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)

    def _run_one(provider_name: str, model_id: str, api_key: str) -> tuple[str, list[dict] | None]:
        """Generate for one model, return (model_short, responses | None)."""
        model_short = _normalize_model_id(model_id)
        try:
            responses = generate_for_model(
                provider_name, model_id, api_key, args.temperature
            )
            return model_short, responses
        except Exception as exc:
            logger.error("Failed to generate for %s: %s", model_short, exc)
            return model_short, None

    if parallel:
        # Suppress per-prompt logging when parallel to reduce noise
        logging.getLogger("indicator_gaming").setLevel(logging.WARNING)

        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(_run_one, prov, mid, key): _normalize_model_id(mid)
                for prov, mid, key in jobs
            }

            for future in as_completed(futures):
                model_short = futures[future]
                ms, responses = future.result()

                with _lock:
                    if responses is not None:
                        all_responses.extend(responses)
                        completed_count += 1
                        n_empty = sum(1 for r in responses if not r["response"])
                        status = "OK" if n_empty == 0 else f"OK ({n_empty} empty)"
                        _save_corpus()
                    else:
                        failed_count += 1
                        status = "FAILED"

                    elapsed = time.time() - start_time
                    done = completed_count + failed_count
                    print(
                        f"  [{done}/{len(jobs)}] {ms:30s}  {status}"
                        f"  ({elapsed:.0f}s elapsed)"
                    )
    else:
        # Sequential — verbose per-prompt logging
        for provider_name, model_id, api_key in jobs:
            model_short = _normalize_model_id(model_id)
            logger.info(
                "Generating responses for %s (%s)...",
                model_short, provider_name,
            )

            ms, responses = _run_one(provider_name, model_id, api_key)

            if responses is not None:
                all_responses.extend(responses)
                completed_count += 1
                _save_corpus()
                n_models = len({r["model_short"] for r in all_responses})
                logger.info(
                    "  Saved %d responses. Corpus total: %d responses from %d models.",
                    len(responses), len(all_responses), n_models,
                )
            else:
                failed_count += 1

    # ── Summary ─────────────────────────────────────────────────────────

    elapsed = time.time() - start_time
    total_new = sum(
        1 for r in all_responses
        if r["model_short"] not in existing_models
    )
    n_models = len({r["model_short"] for r in all_responses})

    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Completed:   {completed_count}/{len(jobs)} models")
    if failed_count:
        print(f"  Failed:      {failed_count}")
    print(f"  New responses: {total_new}")
    print(f"  Total corpus:  {len(all_responses)} responses from {n_models} models")
    print(f"  Time:          {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output:        {output_path}")
    print()


if __name__ == "__main__":
    main()
