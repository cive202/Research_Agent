"""Analyst (writer and editor): writes the brief from the findings, then applies the Critic's reviews.

Context = guideline (what to write) + profiles and few-shot examples (how to write) + research findings.
"""
from __future__ import annotations

import re

from .. import config
from ..grounding import derive_sources
from ..llm import ask_json
from ..models import MAX_COMPETITORS, Brief, Critique, Research, WorkflowState
from ..profiles import writing_context
from ..tracing import track
from ..urls import snap_urls

SYSTEM = (
    "You are the Analyst. Write a SHORT research brief using ONLY the provided findings. "
    "Rules: summary 2-4 sentences; at most TWO competitors and only if the findings support it; "
    "3-5 key observations; "
    "EVERY key observation and EVERY competitor note must end with the IDs of the findings that support it, "
    "written like [F2] or [F1][F4]; cite only IDs that exist in the findings, never invent an ID; "
    "put anything not covered by the findings under missing_information (for example market size, "
    "market share or revenue figures when no finding has them). Return competitors_dropped as 0. "
    "If there are only a few findings, say in the summary that the evidence is thin. "
    "Never invent numbers or facts. If the findings are labelled SAMPLE, say so in the summary. "
    + config.EVIDENCE_RULE_WRITE +
    "If a guideline is given, answer its angle for its audience and cover its key questions and "
    "must-include items where the findings allow; its constraints override the profiles. "
    "Follow the writing profiles. Example briefs show the style only: never copy their facts. "
    "Write plain, specific sentences with concrete names and figures from the findings. Avoid vague "
    "generalisations such as 'most companies', em dashes, and these phrases: "
    + ", ".join(f"'{p}'" for p in config.BANNED_PHRASES[:10]) + ". "
    "FOCUS: the brief is about the focus place or company given in the prompt; name it in the summary. "
    "Findings with on_focus=false are BACKGROUND (they do not mention the focus): use them for at most one "
    "observation, worded as general context (for example 'Globally, ...'), and never name a competitor "
    "from background findings alone. If few findings are about the focus, say so plainly in the summary. "
    + config.UNTRUSTED_NOTE
)


def findings_json(research: Research | None) -> str:
    """The evidence the Analyst and Critic need, compact: mode, focus, findings and gaps.

    Retrieved sources, snippets and queries are left out: sources are rebuilt in code from the cited findings,
    and every extra token counts against Groq's per-request limit.
    """
    if research is None:
        return "{}"
    return research.model_dump_json(include={"mode", "focus", "findings", "data_gaps"},
                                    exclude={"findings": {"__all__": {"snippet"}}})


def draft_json(draft: Brief) -> str:
    """A draft without source snippets (the model never needs them)."""
    return draft.model_dump_json(exclude={"sources": {"__all__": {"snippet"}}, "competitors_dropped": True})


def _context(state: WorkflowState, examples: bool = True) -> str:
    guideline = state.guideline.to_prompt() if state.guideline else f"(no guideline; topic: {state.query})"
    profiles = writing_context(include_examples=examples)
    focus = ", ".join(state.research.focus) if state.research and state.research.focus else "(none named)"
    return (
        f"Query: {state.query}\nFocus: {focus}\n\nGuideline:\n{guideline}"
        + (f"\n\nWriting profiles and examples:\n{profiles}" if profiles else "")
        + f"\n\nFindings (JSON):\n{findings_json(state.research)}"
    )


# A figure with a unit: money, a percentage or a count in millions. Used to decide if numbers were found.
_FIGURE = re.compile(r"(\$|US\$|USD|NPR|Rs\.?)\s?\d|\d[\d,.]*\s?(%|percent|million|billion|crore|lakh|mn|bn)\b", re.I)
NO_FIGURES = "No market size, revenue or market share figures were found in the evidence."


def finalize_brief(brief: Brief, research: Research | None) -> Brief:
    """Rules enforced in code after the model answers, so they do not depend on the prompt being obeyed.

    Competitors are capped (also in the Brief model), sources are rebuilt from the cited findings, sample data
    is labelled in the text, and thin or number-free evidence is stated honestly under missing information.
    """
    brief.competitors = brief.competitors[:MAX_COMPETITORS]
    if research is None:
        return brief
    derived = derive_sources(brief, research)
    if derived:
        brief.sources = derived
    if research.mode == "sample" and not brief.summary.upper().startswith("SAMPLE DATA"):
        brief.summary = "SAMPLE DATA (not live research): " + brief.summary
    notes: list[str] = []
    n = len(research.findings)
    if n < config.THIN_EVIDENCE_FINDINGS:
        notes.append(f"Evidence is thin: only {n} finding(s) were available, so conclusions are tentative.")
    if not any(_FIGURE.search(f.claim) for f in research.findings):
        notes.append(NO_FIGURES)
    have = " ".join(brief.missing_information).lower()
    for note in notes:
        covered = "thin" in have if "thin" in note else ("figures" in have or "market size" in have)
        if not covered:
            brief.missing_information.append(note)
    return brief


@track("analyst")
def analyst(state: WorkflowState, revise: bool = False, draft: Brief | None = None,
            critique: Critique | None = None) -> Brief:
    """Write a first draft, or (revise=True) apply `critique` to `draft` (defaults: the latest of each)."""
    # A revision edits an existing draft, so it does not need the few-shot example (saves tokens).
    user = _context(state, examples=not revise)
    if revise:
        draft = draft or (state.drafts[-1] if state.drafts else state.draft)
        critique = critique or state.critique
    if revise and draft and critique:
        issues = "\n".join(
            f"{n}. [{i.rule}/{i.category}] in {i.section}: {i.comment}"
            for n, i in enumerate(sorted(critique.issues, key=lambda i: i.priority), 1)
        )
        user += (
            f"\n\nYour previous draft:\n{draft_json(draft)}"
            f"\n\nCritic issues, most important first:\n{issues or '(none listed)'}"
            f"\n\nCritic summary: {critique.revision_instructions}"
            "\n\nRevise the draft to fix these issues. Fix the earlier ones first; if two issues conflict, "
            "the earlier one wins (guideline beats research beats profiles). Keep everything that was not "
            "criticised. Still use only the findings."
        )
    brief = ask_json(Brief, SYSTEM, user)
    if state.research:
        snap_urls(brief.sources, state.research.sources)
    return finalize_brief(brief, state.research)
