"""Anthropic Claude provider."""

from __future__ import annotations

import anthropic

from indicator_gaming.providers.base import Provider


class AnthropicProvider(Provider):
    """Wrapper around the Anthropic Messages API."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        return self.complete_multiturn(system, [{"role": "user", "content": user}])

    def complete_multiturn(self, system: str, messages: list[dict[str, str]]) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            system=system,
            messages=messages,
        )
        self.last_reasoning = None
        return message.content[0].text
