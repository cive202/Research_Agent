"""Markdown rendering of briefs and evaluation reports."""
from __future__ import annotations

from .models import Brief, EvaluationReport, WorkflowState


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
        lines += [
            "> **Sample data mode:** findings below are fixed, illustrative sample data, "
            "not live research.",
            "",
        ]
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
    if state.critique:
        verdict = "approved" if state.critique.approved else "revision requested and applied"
        lines += ["---", f"*Critic verdict: {verdict}.*"]
        lines += [f"- [{i.category}] {i.section}: {i.comment}" for i in state.critique.issues]
    if state.evaluation:
        lines += ["", evaluation_to_markdown(state.evaluation)]
    return "\n".join(lines)
