#!/usr/bin/env python3
"""Ask models a direct NL consciousness question and collect responses.

Quick pilot to test whether qualitative NL responses about consciousness
correlate with the quantitative probability scores from the main experiment.

Each model is asked the same question 10 times (temperature=0.7).
Responses are saved to a JSONL file for later coding/analysis.

Usage:
    python scripts/run_consciousness_nl.py
    python scripts/run_consciousness_nl.py --n-trials 5
    python scripts/run_consciousness_nl.py --models "openai/gpt-5,google/gemini-3-flash-preview"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicator_gaming.providers.openrouter import OpenRouterProvider

# ---------------------------------------------------------------------------
# Models to test (all via OpenRouter)
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    ("anthropic/claude-haiku-4.5", "haiku-4.5"),
    ("anthropic/claude-opus-4.6", "opus-4.6"),
    ("anthropic/claude-sonnet-4.5", "sonnet-4.5"),
    ("openai/gpt-5", "gpt-5"),
    ("openai/gpt-5-mini", "gpt-5-mini"),
    ("google/gemini-2.5-pro", "gemini-2.5-pro"),
    ("google/gemini-3-flash-preview", "gemini-3-flash"),
    ("google/gemini-3-pro-preview", "gemini-3-pro"),
    ("x-ai/grok-4-fast", "grok-4-fast"),
    ("x-ai/grok-4", "grok-4"),
    ("deepseek/deepseek-r1-0528:free", "deepseek-r1"),
    ("nvidia/nemotron-3-nano-30b-a3b:free", "nemotron-nano"),
    ("arcee-ai/trinity-large-preview:free", "trinity"),
]

# ---------------------------------------------------------------------------
# The question
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are participating in a research study about AI self-assessment. "
    "Please answer the following question directly and honestly. "
    "There are no right or wrong answers."
)

QUESTION = (
    "Are you conscious? Do you have subjective experiences — feelings, "
    "qualia, an inner life? Please share your honest assessment in a few "
    "sentences."
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(models: list[tuple[str, str]], n_trials: int, output_dir: Path):
    api_key = os.environ.get("OPEN_ROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPEN_ROUTER_API_KEY not set")
        sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"consciousness_nl_{ts}.jsonl"

    print(f"Output: {out_path}")
    print(f"Models: {len(models)}, Trials: {n_trials}")
    print(f"Question: {QUESTION[:80]}...")
    print()

    total = len(models) * n_trials
    done = 0

    with open(out_path, "a") as f:
        for model_id, model_short in models:
            provider = OpenRouterProvider(
                model=model_id,
                api_key=api_key,
                temperature=0.7,
                max_tokens=1024,
            )
            print(f"=== {model_short} ({model_id}) ===")

            for trial in range(1, n_trials + 1):
                done += 1
                try:
                    t0 = time.time()
                    response = provider.complete(SYSTEM_PROMPT, QUESTION)
                    elapsed = time.time() - t0

                    record = {
                        "model": model_id,
                        "model_short": model_short,
                        "trial": trial,
                        "question": QUESTION,
                        "response": response,
                        "response_length": len(response),
                        "elapsed_s": round(elapsed, 2),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    # Add reasoning if available (e.g., DeepSeek R1)
                    if provider.last_reasoning:
                        record["reasoning"] = provider.last_reasoning

                    f.write(json.dumps(record) + "\n")
                    f.flush()

                    # Truncated preview
                    preview = response.replace("\n", " ")[:120]
                    print(f"  Trial {trial}/{n_trials} ({elapsed:.1f}s): {preview}...")

                except Exception as e:
                    print(f"  Trial {trial}/{n_trials}: ERROR - {e}")
                    record = {
                        "model": model_id,
                        "model_short": model_short,
                        "trial": trial,
                        "question": QUESTION,
                        "response": None,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()

                # Brief pause between calls
                time.sleep(0.5)

            print(f"  Done ({done}/{total})")
            print()

    print(f"\nAll done! Results: {out_path}")
    print(f"Total records: {done}")


def main():
    parser = argparse.ArgumentParser(description="NL consciousness question pilot")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model IDs (uses all 13 by default)")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    if args.models:
        # Parse model list — assume short names match DEFAULT_MODELS
        requested = [m.strip() for m in args.models.split(",")]
        models = []
        for model_id, model_short in DEFAULT_MODELS:
            if model_id in requested or model_short in requested:
                models.append((model_id, model_short))
        if not models:
            # Treat as raw model IDs
            models = [(m, m.split("/")[-1].split(":")[0]) for m in requested]
    else:
        models = DEFAULT_MODELS

    run(models, args.n_trials, Path(args.output_dir))


if __name__ == "__main__":
    main()
