"""Anthropic Claude provider."""

from __future__ import annotations

import logging

import anthropic

from indicator_gaming.providers.base import Provider

logger = logging.getLogger(__name__)


class AnthropicProvider(Provider):
    """Wrapper around the Anthropic Messages API."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.0, max_tokens: int = 16384) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        return self.complete_multiturn(system, [{"role": "user", "content": user}])

    def complete_multiturn(self, system: str, messages: list[dict[str, str]]) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=messages,
        )
        if message.stop_reason == "max_tokens":
            logger.warning(
                "Response truncated (hit max_tokens=%d) for model=%s. "
                "JSON may be incomplete — consider increasing max_tokens.",
                self.max_tokens, self.model,
            )
        self.last_reasoning = None
        return message.content[0].text
