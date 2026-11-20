"""Deterministic (rule-based) metrics. No LLM calls."""
from __future__ import annotations

import re
from typing import Optional

from .. import config
from ..models import Brief, MetricResult, WorkflowState
from ..urls import domain, normalize_url


def retrieval_check(state: WorkflowState) -> MetricResult:
    r = state.research
    srcs = r.sources if r else []
    doms = {domain(s.url) for s in srcs if s.url}
    empty = sum(1 for s in srcs if not s.snippet.strip())
    problems = []
    if not r or r.mode != "web":
        problems.append("retrieval not live (sample data)")
    if len(srcs) < config.MIN_SOURCES:
        problems.append(f"only {len(srcs)} sources (need {config.MIN_SOURCES})")
    if len(doms) < config.MIN_DOMAINS:
        problems.append(f"only {len(doms)} unique domains (need {config.MIN_DOMAINS})")
    if empty:
        problems.append(f"{empty} sources with empty snippets")
    score = 1.0 if not problems else max(0.0, 1 - 0.34 * len(problems))
    return MetricResult(name="retrieval_check", score=score, passed=not problems,
                        reason=f"{len(srcs)} sources, {len(doms)} domains" + (f"; {'; '.join(problems)}" if problems else ""))


def citation_validity(state: WorkflowState, brief: Optional[Brief] = None) -> MetricResult:
    brief = brief or state.final or state.draft
    retrieved = {normalize_url(s.url) for s in (state.research.sources if state.research else []) if s.url}
    cited = [s for s in (brief.sources if brief else []) if s.url]
    if not cited:
        ok = state.research is not None and state.research.mode == "sample"
        return MetricResult(name="citation_validity", score=1.0 if ok else 0.0, passed=ok,
                            reason="no URLs cited" + (" (sample data)" if ok else ""))
    bad = [s.url for s in cited if normalize_url(s.url) not in retrieved]
    score = 1 - len(bad) / len(cited)
    used = {normalize_url(s.url) for s in cited} & retrieved
    cov = f"{len(used)}/{len(retrieved)} retrieved sources used"
    return MetricResult(name="citation_validity", score=score, passed=not bad,
                        reason=(f"{len(bad)} cited URLs were never retrieved; " if bad else "all cited URLs were retrieved; ") + cov,
                        details=[f"Not retrieved: {u}" for u in bad])


def structure_check(state: WorkflowState, brief: Optional[Brief] = None) -> MetricResult:
    b = brief or state.final or state.draft
    problems = []
    if not b or not b.summary.strip():
        problems.append("summary missing")
    if b and len(b.competitors) > 2:
        problems.append("more than 2 competitors")
    if b and len(b.key_observations) < 3:
        problems.append("fewer than 3 key observations")
    if b and not b.sources:
        problems.append("no sources section")
    if b and not b.missing_information:
        problems.append("missing-information section empty")
    return MetricResult(name="structure_check", score=1.0 if not problems else 0.5, passed=not problems,
                        reason="all sections present" if not problems else "; ".join(problems))


def brief_text(brief: Brief) -> str:
    """All prose in the brief (not source titles or URLs), for style checks."""
    parts = [brief.summary, *(c.note for c in brief.competitors), *brief.key_observations,
             *brief.missing_information]
    return "\n".join(parts)


def style_check(state: WorkflowState, brief: Optional[Brief] = None) -> MetricResult:
    """Informational: flags AI-slop phrases, em dashes and vague generalisations like 'most companies'."""
    b = brief or state.final or state.draft
    text = brief_text(b) if b else ""
    low = text.lower().replace("’", "'")
    hits = [f"banned phrase: '{p}'" for p in config.BANNED_PHRASES if p in low]
    hits += [f"vague generalisation: '{m.group(0)}'" for m in re.finditer(config.VAGUE_GENERALIZATION, low)]
    if "—" in text:
        hits.append("em dash")
    score = max(0.0, 1.0 - 0.25 * len(hits))
    return MetricResult(name="style_check", score=score, passed=not hits,
                        reason="no slop phrases or vague generalisations" if not hits else f"{len(hits)} style flags",
                        details=hits)
