"""OpenRouter provider (OpenAI-compatible API)."""

from __future__ import annotations

import openai

from indicator_gaming.providers.base import Provider


class OpenRouterProvider(Provider):
    """Wrapper around OpenRouter's OpenAI-compatible API."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def complete(self, system: str, user: str) -> str:
        return self.complete_multiturn(system, [{"role": "user", "content": user}])

    def complete_multiturn(self, system: str, messages: list[dict[str, str]]) -> str:
        all_messages = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
            messages=all_messages,
        )
        if not response.choices:
            raise RuntimeError(
                f"OpenRouter returned no choices (model={self.model}). "
                "The model may be overloaded or the response was empty."
            )
        msg = response.choices[0].message
        content = msg.content
        if not content:
            raise RuntimeError(
                f"OpenRouter returned empty content (model={self.model})."
            )

        # Capture reasoning trace if the model provides one (e.g. DeepSeek R1)
        self.last_reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)

        return content
