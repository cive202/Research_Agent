"""Researcher: retrieves documents and extracts sourced findings."""
from __future__ import annotations

import re

from .. import config
from ..focus import ensure_focus, mentions_focus, rank_by_focus
from ..llm import LLMError, ask_json
from ..models import Finding, FindingDraft, Research, ResearchResult, WorkflowState
from ..retrieval import plan_gap_queries, plan_queries, plan_search, select_relevant
from ..search import SearchUnavailable, last_provider, read_urls, web_search
from ..tracing import track
from ..urls import normalize_url, snap_urls
from .sample_data import SAMPLE_RESEARCH

SYSTEM = (
    "You are the Researcher in a market-research team. From ONLY the numbered documents provided, "
    "extract 5-10 key facts (market overview, main players, trends, challenges) relevant to the topic. "
    "Each finding must name the source_title and source_url of the document it came from. "
    "Do not add facts that are not in the documents. If something could not be found "
    "(market size, growth, competitors, dates), list it in data_gaps instead of guessing. "
    "Only list a gap if the documents truly do not answer it. List each fact once: if several documents say "
    "the same thing, write one finding and cite the most authoritative of them. Documents marked BACKGROUND do not mention "
    "the focus: use them only for general context. " + config.EVIDENCE_RULE_RESEARCH + config.UNTRUSTED_NOTE
)


def _retrieve(query: str, extra_queries: list[str] | None = None) -> tuple[list, list[str], list[str]]:
    """Plan queries and focus, search, drop irrelevant documents; re-search once if too few survive.

    `extra_queries` (the guideline's key questions) run first, ahead of the planned ones. Sources are then
    ranked by the focus: on-focus first, at most MAX_BACKGROUND_SOURCES others, labelled background.
    """
    plan = plan_search(query)
    queries, focus = plan.queries, plan.focus
    if extra_queries:
        seen = {q.lower() for q in extra_queries}
        queries = list(extra_queries) + [q for q in queries if q.lower() not in seen]
        queries = ensure_focus(queries[: max(config.MAX_PLANNED_QUERIES, len(extra_queries))], focus)
    found = web_search(query, queries or None)
    kept = select_relevant(query, found)
    if len(kept) < config.MIN_KEPT_SOURCES or len(kept) / len(found) < config.MIN_KEPT_SHARE:
        try:
            retry = ensure_focus(plan_queries(query, avoid=queries), focus)
            seen = {s.url for s in found}
            extra = [s for s in web_search(query, retry or None) if s.url not in seen] if retry else []
        except SearchUnavailable:
            extra = []
        kept += select_relevant(query, extra)
    if not kept:  # filter rejected everything: better weak evidence than none
        kept = found
    return rank_by_focus(kept, focus)[: config.MAX_SOURCES], queries, focus


def claim_key(claim: str) -> str:
    """Findings with the same text share this key, whatever their case, spacing or punctuation."""
    return re.sub(r"[^a-z0-9]+", "", claim.lower())[:200]


def drop_repeated_claims(findings: list[FindingDraft]) -> list[FindingDraft]:
    """Keep the first finding of each claim. The model sometimes copies one fact once per source."""
    seen: set[str] = set()
    out: list[FindingDraft] = []
    for f in findings:
        key = claim_key(f.claim)
        if key and key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def number_findings(findings: list[FindingDraft], sources: list, focus: list[str] | None = None) -> list[Finding]:
    """Drop repeated claims, cap at MAX_FINDINGS, then number F1, F2, ... in code and attach a source snippet.

    De-duplicating before the cap means the cap counts distinct facts, and the IDs have no gaps.
    A finding is BACKGROUND (on_focus=False) when its source is background and the claim itself does not
    mention the focus either.
    """
    by_url = {normalize_url(s.url): s for s in sources if s.url}
    findings = [f if isinstance(f, Finding) else Finding(**f.model_dump())
                for f in drop_repeated_claims(findings)[: config.MAX_FINDINGS]]
    for n, f in enumerate(findings, 1):
        f.id = f"F{n}"
        src = by_url.get(normalize_url(f.source_url))
        if src and not f.snippet:
            f.snippet = src.snippet[: config.FINDING_SNIPPET_CHARS].strip()
        f.on_focus = (src.on_focus if src else True) or mentions_focus([f.claim], focus or [])
    return findings


def _extract(query: str, sources: list, focus: list[str] | None = None) -> ResearchResult:
    docs = "\n\n".join(f"[{i + 1}]{'' if s.on_focus else ' (BACKGROUND)'} {s.title}\nURL: {s.url}\n{s.snippet}"
                       for i, s in enumerate(sources))
    focus_line = f"Focus: {', '.join(focus)}\n" if focus else ""
    result = ask_json(ResearchResult, SYSTEM, f"Research topic: {query}\n{focus_line}\nDocuments:\n{docs}")
    snap_urls(result.findings, sources, "source_url")
    result.findings = number_findings(result.findings, sources, focus)  # now numbered Finding objects
    return result


def _fill_gaps(query: str, sources: list, result: ResearchResult, tried: list[str], focus: list[str],
               max_rounds: int) -> tuple[list, ResearchResult, list[list[str]]]:
    """Search for what is still missing, then re-extract. Stops when nothing new is found."""
    rounds: list[list[str]] = []
    for _ in range(max_rounds):
        if not result.data_gaps or len(sources) >= config.MAX_TOTAL_SOURCES:
            break
        gap_queries = plan_gap_queries(query, result.data_gaps, tried + [q for r in rounds for q in r], focus)
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
            widened = _rank_keeping_seeds(sources + new, focus)[: config.MAX_TOTAL_SOURCES]
            result, sources = _extract(query, widened, focus), widened
        except LLMError:  # the gap round is a bonus: keep the earlier findings rather than fail the run
            break
    return sources, result, rounds


def _with_seeds(seeds: list, sources: list) -> list:
    """Guideline links come first: the user already trusts them, so they always count as on-focus."""
    seen = {normalize_url(s.url) for s in seeds}
    seeds = [s.model_copy(update={"on_focus": True}) for s in seeds]
    return (seeds + [s for s in sources if normalize_url(s.url) not in seen])[: config.MAX_TOTAL_SOURCES]


def _rank_keeping_seeds(sources: list, focus: list[str]) -> list:
    """Re-rank after a gap round; sources already marked on-focus (such as guideline links) stay on-focus."""
    pinned = [s for s in sources if s.on_focus and not mentions_focus([s.title, s.url, s.snippet], focus)]
    keep = {id(s) for s in pinned}
    return pinned + rank_by_focus([s for s in sources if id(s) not in keep], focus)


@track("researcher")
def researcher(state: WorkflowState, use_sample: bool, max_gap_rounds: int | None = None) -> Research:
    """`max_gap_rounds` overrides config.MAX_GAP_ROUNDS (0 turns the gap loop off)."""
    rounds_allowed = config.MAX_GAP_ROUNDS if max_gap_rounds is None else max(0, int(max_gap_rounds))
    if use_sample:
        return SAMPLE_RESEARCH.model_copy(deep=True)
    g = state.guideline
    topic = g.topic if g else state.query
    seeds = read_urls(g.seed_links[: config.MAX_SEED_LINKS]) if g and g.seed_links else []
    try:
        sources, queries, focus = _retrieve(topic, g.key_questions if g else None)
    except SearchUnavailable as e:
        if not seeds:
            res = SAMPLE_RESEARCH.model_copy(deep=True)
            res.data_gaps.insert(0, f"Live web search unavailable ({e}); using sample data.")
            return res
        sources, queries, focus = [], [], []
    sources = _with_seeds(seeds, sources)

    result = _extract(state.query, sources, focus)
    sources, result, rounds = _fill_gaps(state.query, sources, result, queries, focus, rounds_allowed)
    return Research(mode="web", provider=last_provider() or "web", findings=result.findings,
                    data_gaps=result.data_gaps, sources=sources, queries=queries, gap_rounds=rounds, focus=focus)
