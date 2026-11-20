"""Combine individual metrics into an overall RAG verdict."""
from __future__ import annotations

from ..models import EvaluationReport, WorkflowState
from .judges import brief_groundedness, gap_honesty, research_faithfulness, retrieval_relevance
from .rules import citation_validity, retrieval_check, structure_check, style_check

REQUIRED_METRICS = {
    "retrieval_check", "retrieval_relevance", "research_faithfulness",
    "brief_groundedness", "citation_validity", "structure_check",
}


def evaluate(state: WorkflowState, use_ragas: bool = False) -> EvaluationReport:
    live = bool(state.research and state.research.mode == "web")
    metrics = [retrieval_check(state)]
    if live:
        metrics.append(retrieval_relevance(state))
        metrics.append(research_faithfulness(state))
    metrics += [brief_groundedness(state), citation_validity(state), structure_check(state), style_check(state), gap_honesty(state)]
    if use_ragas and live:
        from .ragas_eval import ragas_metrics

        metrics += ragas_metrics(state)
    if not live:
        return EvaluationReport(metrics=metrics, verdict="RAG NOT ACTIVE",
                                failed_checks=["retrieval is not live (sample data) - retrieval was not exercised"])
    failed = [m.name for m in metrics if m.name in REQUIRED_METRICS and not m.passed]
    return EvaluationReport(metrics=metrics, verdict="RAG WORKING" if not failed else "RAG NOT WORKING",
                            failed_checks=failed)
