"""Analyst: writes (and optionally revises) the brief from the findings."""
from __future__ import annotations

import json

from .. import config
from ..llm import ask_json
from ..models import Brief, WorkflowState
from ..urls import snap_urls

SYSTEM = (
    "You are the Analyst. Write a SHORT research brief using ONLY the provided findings. "
    "Rules: summary 2-4 sentences; at most TWO competitors and only if the findings support it; "
    "3-5 key observations; list sources taken from the findings (title and URL); "
    "put anything not covered by the findings under missing_information. "
    "Never invent numbers or facts. If the findings are labelled SAMPLE, say so in the summary. "
    "Write plain, specific sentences with concrete names and figures from the findings. Avoid vague "
    "generalisations such as 'most companies' and avoid these phrases: "
    + ", ".join(f"'{p}'" for p in config.BANNED_PHRASES[:10]) + "."
)


def analyst(state: WorkflowState, revise: bool = False) -> Brief:
    findings = state.research.model_dump(exclude={"sources": {"__all__": {"snippet"}}}) if state.research else {}
    user = f"Query: {state.query}\n\nFindings (JSON):\n{json.dumps(findings, indent=2)}"
    if revise and state.draft and state.critique:
        issues = "\n".join(
            f"{n}. [{i.category}] in {i.section}: {i.comment}"
            for n, i in enumerate(sorted(state.critique.issues, key=lambda i: i.priority), 1)
        )
        user += (
            f"\n\nYour previous draft:\n{state.draft.model_dump_json(indent=2)}"
            f"\n\nCritic issues, most important first:\n{issues or '(none listed)'}"
            f"\n\nCritic summary: {state.critique.revision_instructions}"
            "\n\nRevise the draft to fix these issues. Fix the earlier ones first; if two issues conflict, "
            "the earlier one wins. Keep everything that was not criticised. Still use only the findings."
        )
    brief = ask_json(Brief, SYSTEM, user)
    if state.research:
        snap_urls(brief.sources, state.research.sources)
    return brief
