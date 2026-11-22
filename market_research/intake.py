"""Orchestrator intake: is the input a company or market we can research? Clarify or decline early.

The check fails OPEN: if the model is unavailable the run simply continues, so intake can never block a run
that would have worked.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .llm import LLMError, ask_json


class Intake(BaseModel):
    is_valid: bool = Field(description="True if the input names a company, product, industry or market.")
    reason: str = Field(default="", description="Why the input was rejected, one sentence.")
    clarifying_question: str = Field(default="", description="One short question if the input is too vague.")


SYSTEM = (
    "You check the input of a market-research app. Accept anything that names a company, product, industry or "
    "market in any country, including short names and instructions such as 'Analyze the food delivery market in "
    "Nepal'. Reject only: gibberish, greetings, questions unrelated to business or markets, and requests that are "
    "harmful. If the input is a real topic but far too vague to research (for example just 'market'), set "
    "is_valid=false and ask ONE short clarifying_question. "
    "Also ask (is_valid=false) when a place or company name has several well-known meanings and nothing in the "
    "input says which one: for example a city name that exists in more than one country, such as Hyderabad "
    "(India or Pakistan), or a name like 'Mercury' (planet, car brand, company). Ask which one is meant and "
    "name the options. A country, region or other detail in the input settles it, so then accept. "
    "Otherwise, when unsure, accept."
)


def check_query(query: str) -> None:
    """Raise ValueError with a friendly message when the query should be clarified or declined."""
    try:
        res = ask_json(Intake, SYSTEM, f"Input: {query}", temperature=0.0)
    except LLMError:
        return
    if not res.is_valid:
        if res.clarifying_question:  # a real topic that needs one more detail, e.g. which Bharatpur
            raise ValueError(f"One question before I start: {res.clarifying_question}")
        msg = res.reason or "That does not look like a company or industry."
        raise ValueError(f"Please give a company or industry to research. {msg}")
