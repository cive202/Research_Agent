"""Optional RAGAS metrics (faithfulness, context precision) reported next to the custom ones.

Non-blocking: failures return a skipped metric and never change the RAG verdict.
"""
from __future__ import annotations

from .. import config
from ..models import Brief, MetricResult, WorkflowState


def _brief_text(b: Brief) -> str:
    parts = [b.summary] + [f"{c.name}: {c.note}" for c in b.competitors] + list(b.key_observations)
    return "\n".join(parts)


def _skipped(name: str, why: str) -> MetricResult:
    return MetricResult(name=name, score=0.0, passed=False, reason=f"skipped: {why}")


def ragas_metrics(state: WorkflowState) -> list[MetricResult]:
    """Run RAGAS with the Groq judge model. Returns `ragas_*` MetricResults (informational)."""
    names = ("ragas_faithfulness", "ragas_context_precision")
    brief = state.final or state.draft
    if not (state.research and state.research.sources and brief):
        return [_skipped(n, "no retrieved documents or brief") for n in names]
    try:
        import warnings

        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

        from ..llm import chat_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            llm = LangchainLLMWrapper(chat_model(config.judge_model_name(), temperature=0.0, max_tokens=8000))
            data = EvaluationDataset.from_list([{
                "user_input": state.query,
                "retrieved_contexts": [s.snippet for s in state.research.sources],
                "response": _brief_text(brief),
            }])
            result = evaluate(data, metrics=[Faithfulness(), LLMContextPrecisionWithoutReference()],
                              llm=llm, show_progress=False, raise_exceptions=False)
        row = result.to_pandas().iloc[0].to_dict()
    except Exception as e:  # ragas/LLM/version problems must not break the workflow
        return [_skipped(n, f"{type(e).__name__}: {e}"[:200]) for n in names]

    out = []
    for name, col in zip(names, ("faithfulness", "llm_context_precision_without_reference")):
        v = row.get(col)
        if v is None or v != v:  # missing or NaN
            out.append(_skipped(name, "RAGAS returned no score"))
        else:
            score = max(0.0, min(1.0, float(v)))
            out.append(MetricResult(name=name, score=score, passed=score >= 0.5,
                                    reason="RAGAS (informational, not part of the verdict)"))
    return out
