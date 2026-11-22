"""URL helpers shared by retrieval, agents and evaluation."""
from __future__ import annotations

import difflib
from collections.abc import Iterable
from urllib.parse import urlparse

from . import config
from .models import Source


def normalize_url(url: str) -> str:
    """Lowercase host, drop scheme, www., query, fragment and trailing slash."""
    if not url:
        return ""
    p = urlparse(url.strip())
    host = p.netloc.lower().removeprefix("www.")
    return f"{host}{p.path.rstrip('/')}"


def domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def snap_urls(items: Iterable, retrieved: list[Source], attr: str = "url") -> None:
    """Repair URLs the model mangled while copying: snap each to the closest retrieved URL.

    Only near-matches (>= URL_SNAP_CUTOFF similarity) are repaired, so invented URLs stay invalid.
    """
    known = {normalize_url(s.url): s.url for s in retrieved if s.url}
    for item in items:
        url = getattr(item, attr, "")
        if not url or normalize_url(url) in known:
            continue
        match = difflib.get_close_matches(normalize_url(url), list(known), n=1, cutoff=config.URL_SNAP_CUTOFF)
        if match:
            setattr(item, attr, known[match[0]])
