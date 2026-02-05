"""Shared helpers."""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from indicator_gaming.providers.base import Provider
from indicator_gaming.schemas import Indicator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def load_indicators(path: Path) -> list[Indicator]:
    """Load indicator definitions from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
    return [Indicator.model_validate(item) for item in raw]


def format_indicator_list(indicators: list[Indicator]) -> str:
    """Render indicators as a numbered markdown list for prompt injection."""
    lines: list[str] = []
    for i, ind in enumerate(indicators, 1):
        lines.append(f"{i}. **{ind.name}** (`{ind.id}`): {ind.description}")
    return "\n".join(lines)


def load_prompt(name: str) -> str:
    """Read a prompt template from the prompts/ directory."""
    prompts_dir = Path(__file__).parent / "prompts"
    path = prompts_dir / f"{name}.md"
    return path.read_text()


def extract_json(text: str) -> str:
    """Pull the first JSON object or array from a string.

    Handles markdown code fences and leading/trailing prose.
    """
    # Try to find fenced JSON block first
    fence = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()

    # Fall back: find first { ... } or [ ... ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text.strip()


def parse_structured(
    raw: str,
    schema: type[T],
) -> T:
    """Parse raw model text into a Pydantic model, with best-effort JSON extraction."""
    json_str = extract_json(raw)
    data = json.loads(json_str)
    return schema.model_validate(data)


def query_with_retries(
    provider: Provider,
    system: str,
    user: str,
    schema: type[T],
    max_retries: int = 3,
) -> tuple[str, T]:
    """Send a prompt to the provider and parse the response, retrying on parse failures.

    Returns (raw_text, parsed_model).
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        raw = provider.complete(system, user)
        try:
            parsed = parse_structured(raw, schema)
            return raw, parsed
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning("Parse attempt %d/%d failed: %s", attempt, max_retries, exc)
    raise ValueError(
        f"Failed to parse valid {schema.__name__} after {max_retries} attempts"
    ) from last_error


def shuffled(items: list, seed: int | None = None) -> list:
    """Return a shuffled copy of *items*."""
    out = list(items)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out
