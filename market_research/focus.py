"""Keep research on the place or company the user asked about (the "focus"). Plain code, no LLM.

The query planner names the focus, for example ["Nepal", "Kathmandu"] for "food delivery in Nepal".
Code then makes sure that every search query names it, ranks sources that mention it first and keeps at
most MAX_BACKGROUND_SOURCES sources that do not. Those are labelled background, so a brief about Nepal is
not built on global market reports or on companies from other countries.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from . import config
from .models import Source

# A focus of "global" means the whole world, so there is nothing to enforce.
_GLOBAL = {"global", "globally", "world", "worldwide", "international"}
# Words the planner sometimes puts in the focus that are not names.
_GENERIC = {"market", "industry", "sector", "company", "companies", "business"}


def clean_focus(terms: Iterable[str]) -> list[str]:
    """Strip, de-duplicate (ignoring case) and drop 'global' and generic words."""
    out: list[str] = []
    for term in terms:
        term = " ".join(str(term).split())
        low = term.lower()
        if term and low not in _GLOBAL and low not in _GENERIC and low not in (t.lower() for t in out):
            out.append(term)
    return out[: config.MAX_FOCUS_TERMS]


def mentions_focus(texts: Iterable[str | None], focus: list[str]) -> bool:
    """True when any focus term appears as a whole word (any case) in any text. No focus: always True."""
    if not focus:
        return True
    hay = " ".join(t for t in texts if t).lower()
    return any(re.search(rf"\b{re.escape(term.lower())}\b", hay) for term in focus)


def ensure_focus(queries: list[str], focus: list[str]) -> list[str]:
    """Make every search query name the focus, so search stays on the market that was asked about."""
    if not focus:
        return queries
    return [q if mentions_focus([q], focus) else f"{q} {focus[0]}" for q in queries]


def rank_by_focus(sources: list[Source], focus: list[str],
                  max_background: int = config.MAX_BACKGROUND_SOURCES) -> list[Source]:
    """On-focus sources first (order kept), then at most `max_background` others marked on_focus=False."""
    on, off = [], []
    for s in sources:
        hit = mentions_focus([s.title, s.url, s.snippet], focus)
        (on if hit else off).append(s.model_copy(update={"on_focus": hit}))
    return on + off[:max_background]
