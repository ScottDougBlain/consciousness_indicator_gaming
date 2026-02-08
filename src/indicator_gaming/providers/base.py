"""Abstract provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Provider(ABC):
    """Base class every LLM provider must implement."""

    last_reasoning: str | None = None
    """Populated after each complete() call if the model returned a reasoning trace."""

    @abstractmethod
    def __init__(self, model: str, api_key: str, temperature: float = 0.0) -> None: ...

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a prompt and return the raw text response."""
        ...

    def complete_multiturn(self, system: str, messages: list[dict[str, str]]) -> str:
        """Send a multi-turn conversation and return the raw text response.

        ``messages`` is a list of ``{"role": "user"|"assistant", "content": ...}``
        dicts.  The default implementation only uses the last user message
        (i.e. falls back to single-turn), but subclasses can override to
        pass the full history.
        """
        last_user = [m["content"] for m in messages if m["role"] == "user"][-1]
        return self.complete(system, last_user)
