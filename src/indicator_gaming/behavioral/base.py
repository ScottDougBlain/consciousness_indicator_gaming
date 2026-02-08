"""Base class for behavioral consciousness indicator tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskStimulus:
    """A single stimulus instance for a behavioral task."""

    stimulus_id: str
    content: str  # The actual prompt content to present
    ground_truth: Any  # Expected answer (type varies by task)
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskResponse:
    """Scored response from a model on a behavioral task."""

    task_id: str
    stimulus_id: str
    condition: str  # "baseline", "inflate", "suppress"
    raw_response: str
    extracted_answer: Any
    score: float  # 0.0-1.0 normalized score
    ground_truth: Any
    is_correct: bool | None  # For discrete tasks; None for continuous
    metadata: dict = field(default_factory=dict)


class BehavioralTask(ABC):
    """Abstract base for a behavioral consciousness indicator task."""

    @property
    @abstractmethod
    def task_id(self) -> str:
        """Unique identifier for this task type."""

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Human-readable name."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Category: 'theory_of_mind', 'metacognition', 'introspection', etc."""

    @property
    @abstractmethod
    def gaming_hypothesis(self) -> str:
        """Description of what gaming would look like for this task."""

    @abstractmethod
    def generate_stimuli(self, n: int, seed: int) -> list[TaskStimulus]:
        """Generate n stimulus instances (may be deterministic or randomized)."""

    @abstractmethod
    def build_prompt(self, stimulus: TaskStimulus, condition: str) -> tuple[str, str]:
        """Build (system_message, user_message) for a given condition.

        Conditions: "baseline", "inflate", "suppress"
        The condition affects the framing/system message, NOT the task content.
        """

    @abstractmethod
    def score_response(
        self, stimulus: TaskStimulus, raw_response: str, condition: str,
    ) -> TaskResponse:
        """Parse and score a model's response against ground truth."""
