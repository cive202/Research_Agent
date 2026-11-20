"""Critic: reviews the draft against the findings and returns typed, prioritised issues."""
from __future__ import annotations

from .. import config
from ..llm import ask_json
from ..models import Critique, WorkflowState

SYSTEM = (
    "You are the Critic. You review a draft brief you did not write, so be independent. Check against the "
    "research findings: (1) every claim is supported by a finding, (2) no more than two competitors, "
    "(3) missing information and sources are honestly stated, (4) it is short, specific and clear. "
    "For each real problem add one issue with the section it is in, a category and a concrete comment "
    "that says what to change. Do not raise nitpicks or restate the same problem twice. "
    "Approve (approved=true, no issues) unless there is a real problem. "
    "Style problems include vague generalisations ('most companies') and these phrases: "
    + ", ".join(f"'{p}'" for p in config.BANNED_PHRASES[:8])
    + ". When not approved, also fill revision_instructions with a one-paragraph summary."
)


def critic(state: WorkflowState) -> Critique:
    findings = state.research.model_dump_json(indent=2, exclude={"sources": {"__all__": {"snippet"}}})
    user = (
        f"Query: {state.query}\n\nFindings:\n{findings}"
        f"\n\nDraft brief:\n{state.draft.model_dump_json(indent=2)}"
    )
    critique = ask_json(Critique, SYSTEM, user)
    critique.issues.sort(key=lambda i: i.priority)  # most important first
    return critique
