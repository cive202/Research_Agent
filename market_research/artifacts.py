"""Save the intermediate files of one run, so every handoff can be inspected afterwards.

runs/<timestamp>-<slug>/
  research.md   findings, gaps and the queries that produced them
  draft_v0.md   the Analyst's first draft
  final.md      the final brief (same as draft_v0 when the Critic approved) plus evaluation
  state.json    the full shared state, including the critique and per-step metrics
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .models import Research, WorkflowState
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


def save_run(state: WorkflowState, base_dir: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", state.query.lower()).strip("-")[:40] or "run"
    path = Path(base_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
    path.mkdir(parents=True, exist_ok=True)
    if state.research:
        (path / "research.md").write_text(research_to_markdown(state.query, state.research), encoding="utf-8")
    if state.draft:
        (path / "draft_v0.md").write_text(brief_to_markdown(state, state.draft), encoding="utf-8")
    if state.final:
        (path / "final.md").write_text(brief_to_markdown(state), encoding="utf-8")
    state.run_dir = str(path)
    (path / "state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return str(path)
