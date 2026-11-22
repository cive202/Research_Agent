"""Critic (the reviewer): checks a draft against the guideline, the research and the writing profiles.

It runs in its own LLM call with its own context, so it is not biased towards a draft it wrote itself.
It returns typed issues, each tagged with the rule it breaks, so the editor knows exactly what to fix.
"""
from __future__ import annotations

from .. import config
from ..llm import ask_json
from ..models import Brief, Critique, WorkflowState
from ..profiles import writing_context
from ..tracing import track

SYSTEM = (
    "You are the Critic. You review a draft brief you did not write, so be independent. Check it against "
    "three things, in this order of importance:\n"
    "1. GUIDELINE (if given): does the brief answer the angle, suit the audience, cover the key questions and "
    "must-include items, and respect the constraints? Use rule='guideline'.\n"
    "2. RESEARCH: is every claim supported by a finding, and are gaps and sources stated honestly? "
    "Use rule='research'.\n"
    "3. PROFILES: does it follow the structure, terminology and character profiles? Use rule='structure', "
    "'terminology' or 'character'.\n"
    "Also: no more than two competitors (category competitor_limit). For each real problem add one issue "
    "with its rule, the section it is in, a category and a concrete comment that says what to change. "
    "Do not raise nitpicks or restate the same problem twice. Approve (approved=true, no issues) unless "
    "there is a real problem. Style problems include vague generalisations ('most companies'), em dashes "
    "and these phrases: " + ", ".join(f"'{p}'" for p in config.BANNED_PHRASES[:8])
    + ". When not approved, also fill revision_instructions with a one-paragraph summary."
)


@track("critic")
def critic(state: WorkflowState, draft: Brief | None = None) -> Critique:
    draft = draft or (state.drafts[-1] if state.drafts else state.draft)
    findings = state.research.model_dump_json(indent=2, exclude={"sources": {"__all__": {"snippet"}}})
    guideline = state.guideline.to_prompt() if state.guideline else f"(no guideline; topic: {state.query})"
    profiles = writing_context(include_examples=False)
    user = (
        f"Query: {state.query}\n\nGuideline:\n{guideline}"
        + (f"\n\nWriting profiles:\n{profiles}" if profiles else "")
        + f"\n\nFindings:\n{findings}\n\nDraft brief:\n{draft.model_dump_json(indent=2)}"
    )
    critique = ask_json(Critique, SYSTEM, user)
    critique.issues.sort(key=lambda i: i.priority)  # most important first
    return critique
