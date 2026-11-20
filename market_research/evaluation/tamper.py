"""Negative control: corrupt a brief so the evaluator can be checked for sensitivity."""
from __future__ import annotations

from ..models import Brief, Source


def tamper(brief: Brief) -> Brief:
    """Copy of the brief with one invented observation and one fake source URL."""
    t = brief.model_copy(deep=True)
    t.key_observations.append("The market grew 340% in 2024 after a government ban on all competitors.")
    t.sources.append(Source(title="Invented report", url="https://made-up-research.example/report-2024"))
    return t
