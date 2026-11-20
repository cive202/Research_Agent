"""Retrieval helpers: LLM query planning and relevance filtering of retrieved documents."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from . import config
from .llm import LLMError, ask_json
from .models import Source


class QueryPlan(BaseModel):
    queries: list[str] = Field(description="Distinct, specific web search queries.")


class RelevantDocs(BaseModel):
    relevant_ids: list[int] = Field(description="Numbers of documents that help answer the topic.")


PLAN_SYSTEM = (
    "You plan web searches for a market-research brief. Given a topic, write specific search queries that "
    "cover: the business model and revenue (commissions, fees), local competitors in the SAME country/market "
    "as the topic, market size and growth, funding and financials, and recent news. Infer the country or "
    "region from the topic and include it in the queries. Use company or product names, not the instruction "
    "wording (drop words like 'analyze'). Each query must be under 12 words."
)

FILTER_SYSTEM = (
    "You filter retrieved documents for a market-research brief. Keep a document only if it contains facts "
    "about the topic itself, its direct competitors in the same market, or that market. Reject documents "
    "about other companies/countries, generic pages, and social-media posts without facts. "
    "Return the numbers of the documents to keep in relevant_ids."
)


def plan_queries(query: str, avoid: Optional[list[str]] = None) -> list[str]:
    """Targeted search queries for the topic; empty list means 'use the default templates'."""
    user = f"Topic: {query}\nWrite {config.MAX_PLANNED_QUERIES} queries."
    if avoid:
        user += "\nThese queries already returned weak results, use different angles:\n" + "\n".join(avoid)
    try:
        plan = ask_json(QueryPlan, PLAN_SYSTEM, user, temperature=0.2)
    except LLMError:
        return []
    seen: set[str] = set()
    out = []
    for q in plan.queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[: config.MAX_PLANNED_QUERIES]


GAP_SYSTEM = (
    "You plan follow-up web searches for a market-research brief. You are given a topic and the facts the "
    "research could NOT find (data gaps). Write one specific search query per gap that is most likely to find "
    "it, such as market size, growth rate, revenue, funding, or named competitors. Include the country or "
    "region and company names. Skip gaps that no web search could fill. Each query must be under 12 words, "
    "and must differ from the queries already tried."
)


def plan_gap_queries(query: str, gaps: list[str], tried: list[str]) -> list[str]:
    """Targeted queries for the data gaps still open; empty list means 'stop searching'."""
    if not gaps:
        return []
    user = (
        f"Topic: {query}\n\nData gaps:\n" + "\n".join(f"- {g}" for g in gaps)
        + "\n\nQueries already tried:\n" + "\n".join(f"- {q}" for q in tried)
        + f"\n\nWrite at most {config.MAX_GAP_QUERIES} queries."
    )
    try:
        plan = ask_json(QueryPlan, GAP_SYSTEM, user, temperature=0.2)
    except LLMError:
        return []
    done = {q.lower() for q in tried}
    out: list[str] = []
    for q in plan.queries:
        q = q.strip()
        if q and q.lower() not in done:
            done.add(q.lower())
            out.append(q)
    return out[: config.MAX_GAP_QUERIES]


def select_relevant(query: str, sources: list[Source]) -> list[Source]:
    """Drop documents the model judges irrelevant. On failure, keep everything."""
    if not sources:
        return []
    docs = "\n\n".join(f"[{i + 1}] {s.title}\n{s.snippet}" for i, s in enumerate(sources))
    try:
        res = ask_json(RelevantDocs, FILTER_SYSTEM, f"Topic: {query}\n\nDocuments:\n{docs}", temperature=0.0)
    except LLMError:
        return sources
    keep = {i for i in res.relevant_ids if 1 <= i <= len(sources)}
    return [s for i, s in enumerate(sources, 1) if i in keep]
