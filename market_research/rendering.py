"""Markdown rendering of briefs and evaluation reports."""
from __future__ import annotations

import json

from .models import Brief, EvaluationReport, WorkflowState

SAMPLE_WARNING = ("SAMPLE DATA MODE: the findings are fixed, illustrative sample data, not live research. "
                  "Do not treat this brief as real market evidence.")


def evidence_to_markdown(state: WorkflowState) -> list[str]:
    """Numbered findings, so every [F#] citation in the brief can be checked."""
    if not state.research:
        return []
    lines = ["## Evidence (findings the brief may cite)"]
    for f in state.research.findings:
        src = f"[{f.source_title or f.source_url}]({f.source_url})" if f.source_url else (f.source_title or "no source")
        lines.append(f"- **{f.id}** {f.claim} ({src})")
    return lines + [""]


def state_to_json(state: WorkflowState) -> str:
    """Full run as JSON, with an explicit data_mode and warning so sample runs cannot be mistaken for real ones."""
    sample = bool(state.research and state.research.mode == "sample")
    payload = {"data_mode": "sample" if sample else "web",
               "warning": SAMPLE_WARNING if sample else None,
               "run": state.model_dump(mode="json")}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def evaluation_to_markdown(ev: EvaluationReport) -> str:
    lines = ["## Evaluation", f"**Verdict: {ev.verdict}**", "",
             "| Metric | Score | Pass | Reason |", "|---|---|---|---|"]
    for m in ev.metrics:
        lines.append(f"| {m.name} | {m.score:.2f} | {'yes' if m.passed else 'no'} | {m.reason} |")
    if ev.failed_checks:
        lines += ["", "Failed checks: " + ", ".join(ev.failed_checks)]
    for m in ev.metrics:
        for d in m.details:
            lines.append(f"- ({m.name}) {d}")
    return "\n".join(lines)


def brief_to_markdown(state: WorkflowState, brief: Brief | None = None) -> str:
    """Render a brief. With `brief` given (e.g. the first draft) only the brief itself is rendered."""
    only_brief = brief is not None
    b = brief or state.final or state.draft
    if b is None:
        return ""
    lines = [f"# Research brief: {state.query}", ""]
    if state.research and state.research.mode == "sample":
        lines += [f"> **{SAMPLE_WARNING}**", ""]
    lines += ["## Summary", b.summary, ""]
    if b.competitors:
        lines.append("## Competitors")
        lines += [f"- **{c.name}**: {c.note}" for c in b.competitors]
        lines.append("")
    lines.append("## Key observations")
    lines += [f"- {o}" for o in b.key_observations]
    lines.append("")
    lines.append("## Sources")
    if b.sources:
        for s in b.sources:
            lines.append(f"- [{s.title}]({s.url})" if s.url else f"- {s.title}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Missing information")
    lines += [f"- {m}" for m in b.missing_information] or ["- None noted"]
    lines.append("")
    if only_brief:
        return "\n".join(lines)
    lines += evidence_to_markdown(state)
    final_flags = state.grounding
    lines += ["## Grounding check"] + (
        [f"- [{f.code}] {f.section}: {f.message}" for f in final_flags] if final_flags
        else ["- No grounding flags: every observation and competitor cites an existing finding."]) + [""]
    if state.critique:
        rounds = len(state.critiques) or 1
        if state.critique.approved:
            verdict = f"approved after {rounds} review round(s)"
        else:
            verdict = f"changes requested in all {rounds} round(s); last edits applied without a further review"
        lines += ["---", f"*Critic verdict: {verdict}. Versions: {max(len(state.drafts), 1)}.*"]
        lines += [f"- [{i.rule}/{i.category}] {i.section}: {i.comment}" for i in state.critique.issues]
    if state.evaluation:
        lines += ["", evaluation_to_markdown(state.evaluation)]
    return "\n".join(lines)
