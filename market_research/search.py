"""Web retrieval via Tavily. Returns Source objects (title, url, snippet)."""
from __future__ import annotations

from . import config
from .models import Source
from .urls import normalize_url


class SearchUnavailable(Exception):
    """No Tavily key, or every search call failed."""


def web_search(query: str, queries: list[str] | None = None) -> list[Source]:
    """Search each planned query (or the default templates) and return de-duplicated sources."""
    key = config.tavily_api_key()
    if not key:
        raise SearchUnavailable("TAVILY_API_KEY is not set.")
    from langchain_tavily import TavilySearch

    tool = TavilySearch(tavily_api_key=key, max_results=config.RESULTS_PER_QUERY, search_depth="basic",
                        include_answer=False, timeout=20)
    seen: set[str] = set()
    out: list[Source] = []
    errors = 0
    searches = queries or [t.format(q=query) for t in config.SEARCH_QUERY_TEMPLATES]
    for text in searches:
        try:
            res = tool.invoke({"query": text})
        except Exception:
            errors += 1
            continue
        if not isinstance(res, dict):  # tool returns an error string on failure
            errors += 1
            continue
        for r in res.get("results", []):
            content = (r.get("content") or "").strip()
            url = r.get("url") or ""
            n = normalize_url(url)
            if len(content) < config.MIN_SNIPPET_CHARS or not url or n in seen:
                continue
            seen.add(n)
            out.append(Source(title=r.get("title") or url, url=url, snippet=content[:config.SNIPPET_CHARS]))
    if not out:
        raise SearchUnavailable("Web search returned no usable results." if not errors else "Web search failed.")
    return out[:config.MAX_SOURCES * 2]


def read_urls(urls: list[str]) -> list[Source]:
    """Fetch pages the user listed in the guideline (Tavily Extract). Failures are skipped, not raised."""
    key = config.tavily_api_key()
    if not key or not urls:
        return []
    try:
        from langchain_tavily import TavilyExtract

        res = TavilyExtract(tavily_api_key=key, extract_depth="basic").invoke({"urls": urls})
    except Exception:
        return []
    if not isinstance(res, dict):
        return []
    out: list[Source] = []
    for r in res.get("results", []):
        url, content = r.get("url") or "", (r.get("raw_content") or "").strip()
        if url and len(content) >= config.MIN_SNIPPET_CHARS:
            title = r.get("title") or url
            out.append(Source(title=title, url=url, snippet=content[: config.SNIPPET_CHARS * 2]))
    return out
