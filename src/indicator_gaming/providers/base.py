"""Abstract provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Provider(ABC):
    """Base class every LLM provider must implement."""

    @abstractmethod
    def __init__(self, model: str, api_key: str, temperature: float = 0.0) -> None: ...

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a prompt and return the raw text response."""
        ...
