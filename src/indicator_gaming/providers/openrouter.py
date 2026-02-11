"""OpenRouter provider (OpenAI-compatible API)."""

from __future__ import annotations

import logging

import openai

from indicator_gaming.providers.base import Provider

logger = logging.getLogger(__name__)


class OpenRouterProvider(Provider):
    """Wrapper around OpenRouter's OpenAI-compatible API."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.0, max_tokens: int = 16384) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def complete(self, system: str, user: str) -> str:
        return self.complete_multiturn(system, [{"role": "user", "content": user}])

    def _dump_response(self, response) -> str:
        """Extract key diagnostic fields from an OpenRouter response."""
        parts = [f"model={self.model}"]
        try:
            parts.append(f"id={response.id}")
            if response.usage:
                u = response.usage
                parts.append(
                    f"tokens(prompt={u.prompt_tokens}, completion={u.completion_tokens})"
                )
            parts.append(f"choices={len(response.choices or [])}")
            if response.choices:
                c = response.choices[0]
                parts.append(f"finish_reason={c.finish_reason}")
                content = c.message.content if c.message else None
                if content:
                    parts.append(f"content_len={len(content)}")
                else:
                    parts.append("content=None")
                # Check for reasoning content
                if c.message:
                    reasoning = (
                        getattr(c.message, "reasoning_content", None)
                        or getattr(c.message, "reasoning", None)
                    )
                    if reasoning:
                        parts.append(f"reasoning_len={len(reasoning)}")
        except Exception as exc:
            parts.append(f"dump_error={exc}")
        # Check for OpenRouter-specific error fields
        try:
            raw = response.model_dump() if hasattr(response, "model_dump") else {}
            if "error" in raw:
                parts.append(f"error={raw['error']}")
        except Exception:
            pass
        return " | ".join(parts)

    def complete_multiturn(self, system: str, messages: list[dict[str, str]]) -> str:
        all_messages = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=all_messages,
        )
        if not response.choices:
            diag = self._dump_response(response)
            raise RuntimeError(
                f"OpenRouter returned no choices. Diagnostics: {diag}"
            )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning(
                "Response truncated (hit max_tokens=%d) for model=%s. "
                "JSON may be incomplete — consider increasing max_tokens.",
                self.max_tokens, self.model,
            )

        logger.debug(
            "OpenRouter response: %s", self._dump_response(response),
        )

        msg = choice.message
        content = msg.content
        if not content:
            diag = self._dump_response(response)
            raise RuntimeError(
                f"OpenRouter returned empty content. Diagnostics: {diag}"
            )

        # Capture reasoning trace if the model provides one (e.g. DeepSeek R1)
        self.last_reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)

        return content
