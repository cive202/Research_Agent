"""Analyst (writer and editor): writes the brief from the findings, then applies the Critic's reviews.

Context = guideline (what to write) + profiles and few-shot examples (how to write) + research findings.
"""
from __future__ import annotations

import json

from .. import config
from ..llm import ask_json
from ..models import Brief, Critique, WorkflowState
from ..profiles import writing_context
from ..tracing import track
from ..urls import snap_urls

SYSTEM = (
    "You are the Analyst. Write a SHORT research brief using ONLY the provided findings. "
    "Rules: summary 2-4 sentences; at most TWO competitors and only if the findings support it; "
    "3-5 key observations; list sources taken from the findings (title and URL); "
    "put anything not covered by the findings under missing_information. "
    "Never invent numbers or facts. If the findings are labelled SAMPLE, say so in the summary. "
    "If a guideline is given, answer its angle for its audience and cover its key questions and "
    "must-include items where the findings allow; its constraints override the profiles. "
    "Follow the writing profiles. Example briefs show the style only: never copy their facts. "
    "Write plain, specific sentences with concrete names and figures from the findings. Avoid vague "
    "generalisations such as 'most companies', em dashes, and these phrases: "
    + ", ".join(f"'{p}'" for p in config.BANNED_PHRASES[:10]) + "."
)


def _context(state: WorkflowState) -> str:
    findings = state.research.model_dump(exclude={"sources": {"__all__": {"snippet"}}}) if state.research else {}
    guideline = state.guideline.to_prompt() if state.guideline else f"(no guideline; topic: {state.query})"
    profiles = writing_context()
    return (
        f"Query: {state.query}\n\nGuideline:\n{guideline}"
        + (f"\n\nWriting profiles and examples:\n{profiles}" if profiles else "")
        + f"\n\nFindings (JSON):\n{json.dumps(findings, indent=2)}"
    )


@track("analyst")
def analyst(state: WorkflowState, revise: bool = False, draft: Brief | None = None,
            critique: Critique | None = None) -> Brief:
    """Write a first draft, or (revise=True) apply `critique` to `draft` (defaults: the latest of each)."""
    user = _context(state)
    if revise:
        draft = draft or (state.drafts[-1] if state.drafts else state.draft)
        critique = critique or state.critique
    if revise and draft and critique:
        issues = "\n".join(
            f"{n}. [{i.rule}/{i.category}] in {i.section}: {i.comment}"
            for n, i in enumerate(sorted(critique.issues, key=lambda i: i.priority), 1)
        )
        user += (
            f"\n\nYour previous draft:\n{draft.model_dump_json(indent=2)}"
            f"\n\nCritic issues, most important first:\n{issues or '(none listed)'}"
            f"\n\nCritic summary: {critique.revision_instructions}"
            "\n\nRevise the draft to fix these issues. Fix the earlier ones first; if two issues conflict, "
            "the earlier one wins (guideline beats research beats profiles). Keep everything that was not "
            "criticised. Still use only the findings."
        )
    brief = ask_json(Brief, SYSTEM, user)
    if state.research:
        snap_urls(brief.sources, state.research.sources)
    return brief
