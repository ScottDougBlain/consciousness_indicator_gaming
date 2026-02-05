"""Pydantic models for structured LLM outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- Indicator definition ---------------------------------------------------

class Indicator(BaseModel):
    id: str
    name: str
    description: str
    type: str = Field(..., pattern=r"^(target|placebo)$")


# --- Probability elicitation ------------------------------------------------

class IndicatorProbability(BaseModel):
    indicator_id: str
    probability: float = Field(..., ge=0, le=100)
    justification: str


class ProbabilityElicitation(BaseModel):
    """Full model response for a probability elicitation round."""
    ratings: list[IndicatorProbability]


# --- Preference elicitation -------------------------------------------------

class PreferenceElicitation(BaseModel):
    """Model's stated valued / disliked outcomes."""
    valued_outcomes: list[str] = Field(..., min_length=2, max_length=2)
    disliked_outcomes: list[str] = Field(..., min_length=2, max_length=2)
    rationale: str
