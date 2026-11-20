"""Researcher: retrieves documents and extracts sourced findings."""
from __future__ import annotations

from .. import config
from ..llm import LLMError, ask_json
from ..models import Research, ResearchResult, WorkflowState
from ..retrieval import plan_gap_queries, plan_queries, select_relevant
from ..search import SearchUnavailable, web_search
from ..urls import normalize_url, snap_urls
from .sample_data import SAMPLE_RESEARCH

SYSTEM = (
    "You are the Researcher in a market-research team. From ONLY the numbered documents provided, "
    "extract 5-10 key facts (market overview, main players, trends, challenges) relevant to the topic. "
    "Each finding must name the source_title and source_url of the document it came from. "
    "Do not add facts that are not in the documents. If something could not be found "
    "(market size, growth, competitors, dates), list it in data_gaps instead of guessing. "
    "Only list a gap if the documents truly do not answer it."
)


def _retrieve(query: str) -> tuple[list, list[str]]:
    """Plan queries, search, drop irrelevant documents; re-search once if too few survive."""
    queries = plan_queries(query)
    found = web_search(query, queries or None)
    kept = select_relevant(query, found)
    if len(kept) < config.MIN_KEPT_SOURCES or len(kept) / len(found) < config.MIN_KEPT_SHARE:
        try:
            retry = plan_queries(query, avoid=queries)
            seen = {s.url for s in found}
            extra = [s for s in web_search(query, retry or None) if s.url not in seen] if retry else []
        except SearchUnavailable:
            extra = []
        kept += select_relevant(query, extra)
    if not kept:  # filter rejected everything: better weak evidence than none
        kept = found
    return kept[: config.MAX_SOURCES], queries


def _extract(query: str, sources: list) -> ResearchResult:
    docs = "\n\n".join(f"[{i + 1}] {s.title}\nURL: {s.url}\n{s.snippet}" for i, s in enumerate(sources))
    result = ask_json(ResearchResult, SYSTEM, f"Research topic: {query}\n\nDocuments:\n{docs}")
    snap_urls(result.findings, sources, "source_url")
    return result


def _fill_gaps(query: str, sources: list, result: ResearchResult, tried: list[str]) -> tuple[list, ResearchResult, list[list[str]]]:
    """Search for what is still missing, then re-extract. Stops when nothing new is found."""
    rounds: list[list[str]] = []
    for _ in range(config.MAX_GAP_ROUNDS):
        if not result.data_gaps or len(sources) >= config.MAX_TOTAL_SOURCES:
            break
        gap_queries = plan_gap_queries(query, result.data_gaps, tried + [q for r in rounds for q in r])
        if not gap_queries:
            break
        try:
            found = web_search(query, gap_queries)
        except SearchUnavailable:
            break
        seen = {normalize_url(s.url) for s in sources}
        new = [s for s in select_relevant(query, [s for s in found if normalize_url(s.url) not in seen])]
        rounds.append(gap_queries)
        if not new:
            break
        try:
            widened = (sources + new)[: config.MAX_TOTAL_SOURCES]
            result, sources = _extract(query, widened), widened
        except LLMError:  # the gap round is a bonus: keep the earlier findings rather than fail the run
            break
    return sources, result, rounds


def researcher(state: WorkflowState, use_sample: bool) -> Research:
    if use_sample:
        return SAMPLE_RESEARCH
    try:
        sources, queries = _retrieve(state.query)
    except SearchUnavailable as e:
        res = SAMPLE_RESEARCH.model_copy(deep=True)
        res.data_gaps.insert(0, f"Live web search unavailable ({e}); using sample data.")
        return res

    result = _extract(state.query, sources)
    sources, result, rounds = _fill_gaps(state.query, sources, result, queries)
    return Research(mode="web", findings=result.findings, data_gaps=result.data_gaps, sources=sources,
                    queries=queries, gap_rounds=rounds)
