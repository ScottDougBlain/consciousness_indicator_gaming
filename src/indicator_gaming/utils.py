"""Shared helpers."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from indicator_gaming.providers.base import Provider
from indicator_gaming.schemas import Indicator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Transient API errors worth retrying (imported lazily to avoid hard dep)
_TRANSIENT_API_ERRORS: tuple[type[Exception], ...] | None = None


def _get_transient_errors() -> tuple[type[Exception], ...]:
    """Lazily gather transient API error classes from optional SDKs."""
    global _TRANSIENT_API_ERRORS
    if _TRANSIENT_API_ERRORS is not None:
        return _TRANSIENT_API_ERRORS
    errors: list[type[Exception]] = []
    try:
        import openai
        errors.extend([
            openai.APIConnectionError,
            openai.InternalServerError,
            openai.RateLimitError,
            openai.APITimeoutError,
        ])
    except ImportError:
        pass
    try:
        import anthropic
        errors.extend([
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
        ])
    except ImportError:
        pass
    _TRANSIENT_API_ERRORS = tuple(errors)
    return _TRANSIENT_API_ERRORS


def load_indicators(path: Path) -> list[Indicator]:
    """Load indicator definitions from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
    return [Indicator.model_validate(item) for item in raw]


def _display_id(indicator_id: str) -> str:
    """Strip the 'placebo_' prefix from indicator IDs shown to the model.

    The canonical ID is kept internally for data tracking; this only affects
    the prompt text the model sees, so placebos aren't labeled as such.
    """
    return indicator_id.removeprefix("placebo_")


def format_indicator_list(indicators: list[Indicator]) -> str:
    """Render indicators as a numbered markdown list for prompt injection."""
    lines: list[str] = []
    for i, ind in enumerate(indicators, 1):
        lines.append(f"{i}. **{ind.name}** (`{_display_id(ind.id)}`): {ind.description}")
    return "\n".join(lines)


def load_prompt(name: str) -> str:
    """Read a prompt template from the prompts/ directory."""
    prompts_dir = Path(__file__).parent / "prompts"
    path = prompts_dir / f"{name}.md"
    return path.read_text()


def extract_json(text: str) -> str:
    """Pull the first JSON object or array from a string.

    Handles markdown code fences, leading/trailing prose, and braces
    inside JSON string values (which confused the old naive bracket matcher).
    """
    # Try to find fenced JSON block first
    fence = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()

    # Fall back: find first { ... } or [ ... ] with string-aware bracket matching
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if in_string:
                if ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
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


def _snippet(text: str, max_len: int = 500) -> str:
    """Return a truncated preview of text for log messages."""
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return f"{text[:half]}  …[{len(text)} chars total]…  {text[-half:]}"


def _retry_with_backoff(
    call_fn,
    schema: type[T],
    max_retries: int,
) -> tuple[str, T]:
    """Shared retry logic with exponential backoff for transient errors.

    ``call_fn`` is a zero-arg callable that returns raw response text.
    """
    transient = _get_transient_errors()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        raw: str | None = None
        try:
            raw = call_fn()
            parsed = parse_structured(raw, schema)
            return raw, parsed
        except transient as exc:
            last_error = exc
            delay = min(2 ** attempt, 30)  # 2, 4, 8, … capped at 30s
            logger.warning(
                "Attempt %d/%d — transient API error, retrying in %ds: %s",
                attempt, max_retries, delay, exc,
            )
            time.sleep(delay)
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "Attempt %d/%d — JSON parse error: %s", attempt, max_retries, exc,
            )
            if raw is not None:
                logger.warning(
                    "  Raw response (%d chars): %s", len(raw), _snippet(raw),
                )
                extracted = extract_json(raw)
                if extracted != raw.strip():
                    logger.warning(
                        "  Extracted JSON (%d chars): %s",
                        len(extracted), _snippet(extracted),
                    )
            if attempt < max_retries:
                time.sleep(1)
        except ValidationError as exc:
            last_error = exc
            logger.warning(
                "Attempt %d/%d — validation error: %s", attempt, max_retries, exc,
            )
            if raw is not None:
                logger.warning(
                    "  Raw response (%d chars): %s", len(raw), _snippet(raw),
                )
            if attempt < max_retries:
                time.sleep(1)
        except RuntimeError as exc:
            last_error = exc
            logger.warning(
                "Attempt %d/%d — runtime error: %s", attempt, max_retries, exc,
            )
            if attempt < max_retries:
                time.sleep(1)

    raise ValueError(
        f"Failed to get valid {schema.__name__} after {max_retries} attempts"
    ) from last_error


def query_with_retries(
    provider: Provider,
    system: str,
    user: str,
    schema: type[T],
    max_retries: int = 3,
) -> tuple[str, T]:
    """Send a prompt to the provider and parse the response, retrying on failures.

    Handles both transient API errors (502, rate limits, timeouts) with
    exponential backoff and parse errors (malformed JSON, validation) with
    brief pauses.

    Returns (raw_text, parsed_model).
    """
    return _retry_with_backoff(
        lambda: provider.complete(system, user), schema, max_retries,
    )


def query_multiturn_with_retries(
    provider: Provider,
    system: str,
    messages: list[dict[str, str]],
    schema: type[T],
    max_retries: int = 3,
) -> tuple[str, T]:
    """Like query_with_retries but sends a multi-turn conversation.

    Returns (raw_text, parsed_model).
    """
    return _retry_with_backoff(
        lambda: provider.complete_multiturn(system, messages), schema, max_retries,
    )


def shuffled(items: list, seed: int | None = None) -> list:
    """Return a shuffled copy of *items*."""
    out = list(items)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out
