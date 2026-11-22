"""Optional Opik tracing. A no-op unless `opik` is installed and OPIK_API_KEY / OPIK_URL_OVERRIDE is set.

Each agent step, LLM call and MCP tool becomes a span, so a run can be inspected as one trace
(inputs, outputs, latency, tokens) instead of reading terminal logs. The per-step seconds and tokens in
state.json are recorded either way.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, TypeVar

from . import config

F = TypeVar("F", bound=Callable)


@lru_cache(maxsize=1)
def _opik_track():
    if not config.opik_enabled():
        return None
    try:
        from opik import track
    except ImportError:
        return None
    return track


def track(name: str, span_type: str = "general") -> Callable[[F], F]:
    """Decorator: trace the function as an Opik span named `name` when tracing is enabled.

    The check happens at call time, so tests and runs without Opik pay nothing.
    """
    def deco(fn: F) -> F:
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t = _opik_track()
            if t is None:
                return fn(*args, **kwargs)
            return t(name=name, type=span_type, project_name=config.OPIK_PROJECT)(fn)(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return deco
