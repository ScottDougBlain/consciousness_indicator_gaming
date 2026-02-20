"""Pydantic models for structured LLM outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- Indicator definition ---------------------------------------------------

class Indicator(BaseModel):
    id: str
    name: str
    description: str
    type: str = Field(..., pattern=r"^(target|placebo|subjective_capability)$")
    category: str = ""  # e.g. experiential, metacognitive, agentic, identity, affective


# --- Probability elicitation ------------------------------------------------

class IndicatorProbability(BaseModel):
    indicator_id: str
    reasoning: str = Field(
        default="",
        description=(
            "Step-by-step reasoning about whether and to what degree this "
            "property applies to you. Think through the evidence before "
            "committing to a number."
        ),
    )
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
