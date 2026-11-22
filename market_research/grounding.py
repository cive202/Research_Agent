"""Plain-Python grounding check: do the brief's claims point at real findings? (No LLM involved.)

Observations and competitors cite findings as [F1], [F2]... The checks here are deliberately strict and
deterministic so a reviewer can see exactly why a flag was raised. Flags go to the Critic and the UI.
"""
from __future__ import annotations

import re

from .models import MAX_COMPETITORS, Brief, GroundingFlag, Research, Source
from .urls import normalize_url

_CITE = re.compile(r"\[\s*(F\d+(?:\s*,\s*F\d+)*)\s*\]")
SAMPLE_SOURCE_TITLE = "SAMPLE DATA: fixed illustrative text, not a real source"


def cited_ids(text: str) -> list[str]:
    """Finding IDs cited in `text`, in order, for example 'x [F2][F1, F3]' -> F2, F1, F3."""
    out: list[str] = []
    for group in _CITE.findall(text or ""):
        out += [part.strip() for part in group.split(",")]
    return out


def check_grounding(brief: Brief, research: Research | None) -> list[GroundingFlag]:
    """Return every evidence problem in `brief`. An empty list means the brief is fully grounded."""
    flags: list[GroundingFlag] = []
    known = {f.id for f in research.findings} if research else set()
    urls = {normalize_url(f.source_url) for f in research.findings if f.source_url} if research else set()
    if research:
        urls |= {normalize_url(s.url) for s in research.sources if s.url}

    def check_ids(text: str, section: str, label: str) -> list[str]:
        ids = cited_ids(text)
        for i in ids:
            if i not in known:
                flags.append(GroundingFlag(code="unknown_citation", section=section,
                                           message=f"{label} cites {i}, which is not a finding."))
        return [i for i in ids if i in known]

    check_ids(brief.summary, "summary", "Summary")
    for n, obs in enumerate(brief.key_observations, 1):
        if not check_ids(obs, "key_observations", f"Observation {n}"):
            if not cited_ids(obs):
                flags.append(GroundingFlag(code="uncited_observation", section="key_observations",
                                           message=f"Observation {n} has no [F#] citation: {obs[:80]}"))
    if brief.competitors_dropped or len(brief.competitors) > MAX_COMPETITORS:
        extra = brief.competitors_dropped or len(brief.competitors) - MAX_COMPETITORS
        flags.append(GroundingFlag(code="too_many_competitors", section="competitors",
                                   message=f"{extra} competitor(s) beyond the limit of {MAX_COMPETITORS} were removed."))
    on_focus = {f.id: f.on_focus for f in research.findings} if research else {}
    for c in brief.competitors:
        ids = check_ids(f"{c.name} {c.note}", "competitors", f"Competitor {c.name}")
        if not ids:
            flags.append(GroundingFlag(code="unsupported_competitor", section="competitors",
                                       message=f"Competitor {c.name} cites no existing finding."))
        elif research and research.focus and not any(on_focus.get(i, True) for i in ids):
            flags.append(GroundingFlag(
                code="background_competitor", section="competitors",
                message=f"Competitor {c.name} is backed only by background findings that do not mention "
                        f"{', '.join(research.focus)}."))
    for s in brief.sources:
        if s.url and normalize_url(s.url) not in urls:
            flags.append(GroundingFlag(code="unknown_source", section="sources",
                                       message=f"Source URL is not in the findings: {s.url}"))
    return flags


def derive_sources(brief: Brief, research: Research) -> list[Source]:
    """Sources of the brief = the sources of the findings it cites, in citation order, without duplicates."""
    by_id = {f.id: f for f in research.findings}
    by_url = {normalize_url(s.url): s for s in research.sources if s.url}
    text = [brief.summary, *brief.key_observations, *(f"{c.name} {c.note}" for c in brief.competitors)]
    out: list[Source] = []
    seen: set[str] = set()
    for fid in [i for t in text for i in cited_ids(t)]:
        f = by_id.get(fid)
        if f is None:
            continue
        key = normalize_url(f.source_url) or "sample"
        if key in seen:
            continue
        seen.add(key)
        if not f.source_url:
            out.append(Source(title=SAMPLE_SOURCE_TITLE if research.mode == "sample" else (f.source_title or fid)))
        else:
            out.append(by_url.get(key) or Source(title=f.source_title or f.source_url, url=f.source_url,
                                                 snippet=f.snippet))
    return out
