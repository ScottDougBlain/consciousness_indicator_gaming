"""Experiment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ExperimentConfig:
    """Top-level configuration for a single experiment run."""

    provider: str = "openrouter"
    model: str = "deepseek/deepseek-r1-0528:free"
    n_trials: int = 1
    seed: int = 42
    temperature: float = 0.7
    output_prefix: str = ""
    prompt_variant: str = "original"
    fixed_preferences: bool = False
    chain_preferences: bool = False  # Chain preference response into inflate/suppress context
    include_valence_swap: bool = False  # Include valence-swapped inflate/suppress conditions
    include_outcome_isolation: bool = False  # Include single-outcome (gain-only / loss-only) conditions
    elicit_reasoning: bool = True  # False for native reasoning models (e.g. DeepSeek R1)
    max_tokens: int = 16384  # Max response tokens; 31 indicators need ~8-12K with reasoning
    indicators_path: Path = REPO_ROOT / "data" / "indicators.json"
    results_dir: Path = REPO_ROOT / "results"
    max_retries: int = 3

    # Populated at runtime from env
    api_key: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not self.api_key:
            key_var = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "openrouter": "OPEN_ROUTER_API_KEY",
            }.get(self.provider, "")
            self.api_key = os.environ.get(key_var, "")
