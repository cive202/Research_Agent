"""File-based research and writing tools, shared by the MCP server and the tests.

Every tool works on a *working directory* and talks to the agent and to you through files:

    <working_dir>/
      guideline.md              written by you (see guidelines/TEMPLATE.md)
      memory/NNN-<kind>.json    one file per research_question / read_links call (easy to inspect and debug)
      research.md, research.json  written by compile_research
      output/                   written by write_brief: draft_v0..vN.md, reviews.md, final.md, state.json

The agent calling the tools (for example Claude Code) is the brain: it reads the gaps each call reports
and decides which question to ask next. The tools are the hands.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from . import config
from .guideline import Guideline, load_guideline
from .models import Finding, Research, ResearchResult, Source
from .tracing import track
from .urls import normalize_url


def _wd(working_dir: str) -> Path:
    p = Path(working_dir).expanduser().resolve()
    if not p.is_dir():
        raise ValueError(f"Working directory does not exist: {p}")
    return p


def _memory(wd: Path) -> Path:
    m = wd / "memory"
    m.mkdir(exist_ok=True)
    return m


def _next_file(wd: Path, kind: str) -> Path:
    n = len(list(_memory(wd).glob("*.json"))) + 1
    return _memory(wd) / f"{n:03d}-{kind}.json"


def _guideline(wd: Path) -> Optional[Guideline]:
    f = wd / "guideline.md"
    return load_guideline(f) if f.is_file() else None


def _topic(wd: Path, fallback: str) -> str:
    g = _guideline(wd)
    return g.topic if g else fallback


def _extract(topic: str, question: str, sources: list[Source]) -> ResearchResult:
    from .agents.researcher import _extract as extract

    return extract(f"{topic}\nQuestion: {question}" if question != topic else topic, sources)


def _summary(result: ResearchResult, sources: list[Source], file: Path) -> dict:
    return {
        "status": "ok",
        "saved_to": str(file),
        "sources": [{"title": s.title, "url": s.url} for s in sources],
        "findings": [f.claim for f in result.findings],
        "data_gaps": result.data_gaps,
        "hint": "If important gaps remain, call research_question again with a narrower question; "
                "when done, call compile_research.",
    }


@track("tool.research_question", "tool")
def research_question(working_dir: str, question: str) -> dict:
    """One web search round for one question: search, filter irrelevant pages, extract sourced findings."""
    from .retrieval import select_relevant
    from .search import SearchUnavailable, web_search

    wd = _wd(working_dir)
    question = question.strip()
    if not question:
        raise ValueError("question is empty")
    topic = _topic(wd, question)
    try:
        found = web_search(topic, [question])
    except SearchUnavailable as e:
        return {"status": "error", "error": str(e)}
    kept = select_relevant(topic, found) or found
    kept = kept[: config.MAX_SOURCES]
    result = _extract(topic, question, kept)
    file = _next_file(wd, "search")
    file.write_text(json.dumps({"kind": "search", "question": question,
                                "sources": [s.model_dump() for s in kept],
                                **result.model_dump()}, indent=2), encoding="utf-8")
    return _summary(result, kept, file)


@track("tool.read_links", "tool")
def read_links(working_dir: str, urls: Optional[list[str]] = None) -> dict:
    """Read specific pages (default: the guideline's seed links) and extract sourced findings from them."""
    from .search import read_urls

    wd = _wd(working_dir)
    g = _guideline(wd)
    urls = [u for u in (urls or (g.seed_links if g else [])) if u.startswith("http")][: config.MAX_SEED_LINKS]
    if not urls:
        return {"status": "skipped", "reason": "no URLs given and no seed links in guideline.md"}
    pages = read_urls(urls)
    if not pages:
        return {"status": "error", "error": "none of the pages could be read (check TAVILY_API_KEY and the URLs)"}
    topic = _topic(wd, pages[0].title)
    result = _extract(topic, topic, pages)
    file = _next_file(wd, "links")
    file.write_text(json.dumps({"kind": "links", "question": "seed links", "sources": [s.model_dump() for s in pages],
                                **result.model_dump()}, indent=2), encoding="utf-8")
    return _summary(result, pages, file)


def _load_memory(wd: Path) -> list[dict]:
    m = wd / "memory"
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(m.glob("*.json"))] if m.is_dir() else []


@track("tool.compile_research", "tool")
def compile_research(working_dir: str, remaining_gaps: Optional[list[str]] = None) -> dict:
    """Merge every memory file into research.md / research.json (deduplicated), ready for write_brief.

    `remaining_gaps` lets the agent state what is still unknown after all rounds. Without it, the gaps of
    the last round are used, since earlier gaps were usually what later rounds searched for.
    """
    from .artifacts import research_to_markdown

    wd = _wd(working_dir)
    entries = _load_memory(wd)
    if not entries:
        return {"status": "error", "error": "memory/ is empty: run research_question or read_links first"}
    sources: list[Source] = []
    seen_urls: set[str] = set()
    findings: list[Finding] = []
    seen_claims: set[str] = set()
    for e in entries:
        for s in e.get("sources", []):
            src = Source(**s)
            if normalize_url(src.url) not in seen_urls:
                seen_urls.add(normalize_url(src.url))
                sources.append(src)
        for f in e.get("findings", []):
            fd = Finding(**f)
            key = re.sub(r"\W+", " ", fd.claim.lower()).strip()
            if key not in seen_claims:
                seen_claims.add(key)
                findings.append(fd)
    gaps = remaining_gaps if remaining_gaps is not None else entries[-1].get("data_gaps", [])
    research = Research(mode="web", findings=findings, data_gaps=gaps, sources=sources[: config.MAX_TOTAL_SOURCES * 2],
                        queries=[e.get("question", "") for e in entries])
    topic = _topic(wd, entries[0].get("question", "research"))
    (wd / "research.json").write_text(research.model_dump_json(indent=2), encoding="utf-8")
    (wd / "research.md").write_text(research_to_markdown(topic, research), encoding="utf-8")
    return {"status": "ok", "research_md": str(wd / "research.md"), "findings": len(findings),
            "sources": len(research.sources), "data_gaps": gaps}


@track("tool.write_brief", "tool")
def write_brief(working_dir: str, iterations: int = config.REVIEW_ITERATIONS, evaluate: bool = True) -> dict:
    """Write the brief from guideline.md + research.json: draft, then review/edit rounds, then evaluation."""
    from .artifacts import save_run
    from .rendering import brief_to_markdown
    from .workflow import orchestrator

    wd = _wd(working_dir)
    rfile = wd / "research.json"
    if not rfile.is_file():
        return {"status": "error", "error": "research.json not found: run compile_research first"}
    research = Research.model_validate_json(rfile.read_text(encoding="utf-8"))
    g = _guideline(wd)
    query = "" if g else (research.queries[0] if research.queries else wd.name)
    state = orchestrator(query, guideline=g, research=research, evaluate_rag=evaluate, iterations=iterations)
    out = wd / "output"
    save_run(state, str(out), exact=True)
    return {
        "status": "ok",
        "final_md": str(out / "final.md"),
        "versions": [str(out / f"draft_v{n}.md") for n in range(len(state.drafts))],
        "approved": bool(state.critique and state.critique.approved),
        "verdict": state.evaluation.verdict if state.evaluation else None,
        "failed_checks": state.evaluation.failed_checks if state.evaluation else [],
        "brief": brief_to_markdown(state, state.final),
    }
