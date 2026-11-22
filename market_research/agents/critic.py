"""Critic (the reviewer): checks a draft against the guideline, the research and the writing profiles.

It runs in its own LLM call with its own context, so it is not biased towards a draft it wrote itself.
It returns typed issues, each tagged with the rule it breaks, so the editor knows exactly what to fix.
"""
from __future__ import annotations

from .. import config
from ..grounding import check_grounding
from ..llm import ask_json
from ..models import Brief, Critique, Issue, WorkflowState
from ..profiles import writing_context
from ..tracing import track
from .analyst import draft_json, findings_json

SYSTEM = (
    "You are the Critic. You review a draft brief you did not write, so be independent. Check it against "
    "three things, in this order of importance:\n"
    "1. GUIDELINE (if given): does the brief answer the angle, suit the audience, cover the key questions and "
    "must-include items, and respect the constraints? Use rule='guideline'.\n"
    "2. RESEARCH: is every claim supported by a finding, and are gaps and sources stated honestly? "
    "Use rule='research'.\n"
    "3. PROFILES: does it follow the structure, terminology and character profiles? Use rule='structure', "
    "'terminology' or 'character'.\n"
    "A list of GROUNDING FLAGS from an automatic check is included: they are facts, not opinions. Turn each "
    "one into an issue (rule='research') that tells the Analyst how to fix it. "
    "Also: no more than two competitors (category competitor_limit). For each real problem add one issue "
    "with its rule, the section it is in, a category and a concrete comment that says what to change. "
    "Do not raise nitpicks or restate the same problem twice. Approve (approved=true, no issues) unless "
    "there is a real problem. Style problems include vague generalisations ('most companies'), em dashes "
    "and these phrases: " + ", ".join(f"'{p}'" for p in config.BANNED_PHRASES[:8])
    + ". FOCUS: the brief must be about the focus given in the prompt, not the world in general: at most "
    "one observation may rest on BACKGROUND findings (on_focus=false), worded as general context, and no "
    "competitor may rest on background findings alone (rule='research'). "
    "When not approved, also fill revision_instructions with a one-paragraph summary. " + config.UNTRUSTED_NOTE
)


@track("critic")
def critic(state: WorkflowState, draft: Brief | None = None) -> Critique:
    draft = draft or (state.drafts[-1] if state.drafts else state.draft)
    findings = findings_json(state.research)
    guideline = state.guideline.to_prompt() if state.guideline else f"(no guideline; topic: {state.query})"
    profiles = writing_context(include_examples=False)
    flags = check_grounding(draft, state.research)
    flag_text = "\n".join(f"- [{f.code}] {f.section}: {f.message}" for f in flags) or "(none)"
    focus = ", ".join(state.research.focus) if state.research and state.research.focus else "(none named)"
    user = (
        f"Query: {state.query}\nFocus: {focus}\n\nGuideline:\n{guideline}"
        + (f"\n\nWriting profiles:\n{profiles}" if profiles else "")
        + f"\n\nFindings:\n{findings}\n\nGrounding flags:\n{flag_text}"
        + f"\n\nDraft brief:\n{draft_json(draft)}"
    )
    critique = ask_json(Critique, SYSTEM, user)
    # The grounding flags are facts, so an approval that ignores them is overruled in code. The competitor
    # cap is excluded: it was already enforced when the brief was built.
    open_flags = [f for f in flags if f.code != "too_many_competitors"]
    if open_flags and critique.approved:
        critique.approved = False
        critique.issues += [
            Issue(rule="research", section=f.section, category="unsupported_claim", comment=f.message)
            for f in open_flags
        ]
        critique.revision_instructions = (
            "The automatic grounding check found problems. Cite existing finding IDs like [F2] on every "
            "observation and competitor, and use only sources from the findings.")
    critique.issues.sort(key=lambda i: i.priority)  # most important first
    return critique
