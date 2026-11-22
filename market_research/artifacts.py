"""Save the intermediate files of one run, so every handoff can be inspected afterwards.

runs/<timestamp>-<slug>/
  research.md   findings, gaps and the queries that produced them
  guideline.md  the guideline the run started from (if any)
  draft_v0.md   the Analyst's first draft; draft_v1.md ... one per review round
  reviews.md    every Critic review, round by round
  final.md      the final brief (the last version) plus evaluation
  state.json    the full shared state, including all reviews and per-step metrics
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .models import Critique, Research, WorkflowState
from .rendering import brief_to_markdown


def research_to_markdown(query: str, r: Research) -> str:
    lines = [f"# Research: {query}", "", f"Mode: {r.mode}", ""]
    if r.queries:
        lines += ["## Initial queries", *[f"- {q}" for q in r.queries], ""]
    for n, qs in enumerate(r.gap_rounds, 1):
        lines += [f"## Gap round {n} queries", *[f"- {q}" for q in qs], ""]
    lines.append("## Findings")
    lines += [f"- {f.claim} ([{f.source_title or 'source'}]({f.source_url}))" if f.source_url else f"- {f.claim}"
              for f in r.findings]
    lines += ["", "## Data gaps", *([f"- {g}" for g in r.data_gaps] or ["- None"]), "", "## Sources"]
    lines += [f"- [{s.title}]({s.url})" for s in r.sources] or ["- None"]
    return "\n".join(lines) + "\n"


def reviews_to_markdown(critiques: list[Critique]) -> str:
    lines = ["# Critic reviews", ""]
    for n, c in enumerate(critiques):
        lines.append(f"## Review of draft_v{n}: {'approved' if c.approved else 'changes requested'}")
        lines += [f"- [{i.rule}/{i.category}] {i.section}: {i.comment}" for i in c.issues] or ["- No issues."]
        if c.revision_instructions:
            lines += ["", f"Summary: {c.revision_instructions}"]
        lines.append("")
    return "\n".join(lines)


def save_run(state: WorkflowState, base_dir: str, exact: bool = False) -> str:
    """Write the run files to a new timestamped folder under `base_dir` (or into `base_dir` if exact)."""
    slug = re.sub(r"[^a-z0-9]+", "-", state.query.lower()).strip("-")[:40] or "run"
    path = Path(base_dir) if exact else Path(base_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
    path.mkdir(parents=True, exist_ok=True)
    if exact:  # rewriting a fixed folder: drop versions left over from an earlier, longer run
        for old in path.glob("draft_v*.md"):
            old.unlink()
    if state.research:
        (path / "research.md").write_text(research_to_markdown(state.query, state.research), encoding="utf-8")
    if state.guideline:
        (path / "guideline.md").write_text(state.guideline.to_prompt() + "\n", encoding="utf-8")
    for n, d in enumerate(state.drafts or ([state.draft] if state.draft else [])):
        (path / f"draft_v{n}.md").write_text(brief_to_markdown(state, d), encoding="utf-8")
    if state.critiques:
        (path / "reviews.md").write_text(reviews_to_markdown(state.critiques), encoding="utf-8")
    if state.final:
        (path / "final.md").write_text(brief_to_markdown(state), encoding="utf-8")
    state.run_dir = str(path)
    (path / "state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return str(path)
