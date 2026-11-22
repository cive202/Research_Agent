"""Web retrieval: Tavily if a key is set, else DuckDuckGo. Returns Source objects (title, url, snippet)."""
from __future__ import annotations

from . import config
from .models import Source
from .urls import normalize_url


class SearchUnavailable(Exception):
    """No search provider is usable, or every search call failed."""


_last_provider = ""


def last_provider() -> str:
    """Name of the provider that produced the most recent results: 'tavily' or 'duckduckgo'."""
    return _last_provider


def _tavily_fetch(key: str):
    from langchain_tavily import TavilySearch

    tool = TavilySearch(tavily_api_key=key, max_results=config.RESULTS_PER_QUERY, search_depth="basic",
                        include_answer=False, timeout=20)

    def fetch(text: str) -> list[dict]:
        res = tool.invoke({"query": text})
        if not isinstance(res, dict):  # the tool returns an error string on failure
            raise RuntimeError(str(res))
        return [{"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
                for r in res.get("results", [])]

    return fetch


def _ddgs_fetch():
    from ddgs import DDGS

    def fetch(text: str) -> list[dict]:
        rows = DDGS(timeout=15).text(text, max_results=config.RESULTS_PER_QUERY)
        return [{"title": r.get("title"), "url": r.get("href"), "content": r.get("body")} for r in rows or []]

    return fetch


def _collect(searches: list[str], fetch, min_chars: int) -> tuple[list[Source], int]:
    """Run every query through `fetch`; return de-duplicated sources and the number of failed queries."""
    seen: set[str] = set()
    out: list[Source] = []
    errors = 0
    for text in searches:
        try:
            rows = fetch(text)
        except Exception:
            errors += 1
            continue
        for r in rows:
            content = (r.get("content") or "").strip()
            url = r.get("url") or ""
            n = normalize_url(url)
            if len(content) < min_chars or not url or n in seen:
                continue
            seen.add(n)
            out.append(Source(title=r.get("title") or url, url=url, snippet=content[:config.SNIPPET_CHARS]))
    return out, errors


def web_search(query: str, queries: list[str] | None = None) -> list[Source]:
    """Search each planned query (or the default templates) and return de-duplicated sources.

    Provider order: Tavily when TAVILY_API_KEY is set, then DuckDuckGo (the `ddgs` package, no key).
    Raises SearchUnavailable when neither works, so the caller can fall back to labelled sample data.
    """
    global _last_provider
    searches = queries or [t.format(q=query) for t in config.SEARCH_QUERY_TEMPLATES]
    providers: list[tuple[str, object, int]] = []
    key = config.tavily_api_key()
    if key:
        providers.append(("tavily", lambda: _tavily_fetch(key), config.MIN_SNIPPET_CHARS))
    providers.append(("duckduckgo", _ddgs_fetch, config.MIN_SNIPPET_CHARS // 2))
    problems: list[str] = []
    for name, make_fetch, min_chars in providers:
        try:
            fetch = make_fetch()
        except ImportError:
            problems.append(f"{name} is not installed")
            continue
        found, errors = _collect(searches, fetch, min_chars)
        if found:
            _last_provider = name
            return found[:config.MAX_SOURCES * 2]
        problems.append(f"{name} failed" if errors else f"{name} returned no usable results")
    raise SearchUnavailable("; ".join(problems) or "No search provider available.")


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
