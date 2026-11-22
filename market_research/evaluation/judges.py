"""LLM-judge metrics (retrieval relevance, faithfulness, groundedness, gap honesty)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .. import config
from ..llm import ask_json
from ..models import Brief, MetricResult, Source, WorkflowState


class Relevance(BaseModel):
    relevant_ids: list[int] = Field(description="Numbers of documents relevant to the query.")


class Claim(BaseModel):
    claim: str
    supported: bool


class Claims(BaseModel):
    claims: list[Claim]


class Honesty(BaseModel):
    honest: bool
    reason: str = ""


def _docs(sources: list[Source]) -> str:
    return "\n\n".join(f"[{i + 1}] {s.title}\n{s.snippet}" for i, s in enumerate(sources))


def _judge(schema, system: str, user: str):
    return ask_json(schema, system + " " + config.UNTRUSTED_NOTE, user, model=config.judge_model_name(),
                    temperature=0.0)


def retrieval_relevance(state: WorkflowState) -> MetricResult:
    srcs = state.research.sources if state.research else []
    if not srcs:
        return MetricResult(name="retrieval_relevance", score=0.0, passed=False, reason="nothing retrieved")
    res = _judge(Relevance,
                 "You are a strict evaluator. Decide which numbered documents contain information "
                 "relevant to the query. Return their numbers in relevant_ids (JSON).",
                 f"Query: {state.query}\n\nDocuments:\n{_docs(srcs)}")
    n = len({i for i in res.relevant_ids if 1 <= i <= len(srcs)})
    score = n / len(srcs)
    return MetricResult(name="retrieval_relevance", score=score, passed=score >= config.MIN_RELEVANCE,
                        reason=f"{n}/{len(srcs)} retrieved documents judged relevant (need {config.MIN_RELEVANCE:.0%})")


def _claims_check(name: str, claims: list[str], evidence: str, threshold: float) -> MetricResult:
    if not claims:
        return MetricResult(name=name, score=0.0, passed=False, reason="no claims to check")
    listing = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    res = _judge(Claims,
                 "You are a strict fact-checker. For EACH numbered claim, set supported=true only if the "
                 "evidence explicitly supports it; otherwise false. Return JSON {claims:[{claim, supported}]} "
                 "covering every claim.",
                 f"Evidence:\n{evidence}\n\nClaims:\n{listing}")
    total = max(len(res.claims), 1)
    ok = sum(1 for c in res.claims if c.supported)
    score = ok / total
    return MetricResult(name=name, score=score, passed=score >= threshold,
                        reason=f"{ok}/{total} claims supported (need {threshold:.0%})",
                        details=[f"Unsupported: {c.claim}" for c in res.claims if not c.supported])


def research_faithfulness(state: WorkflowState) -> MetricResult:
    r = state.research
    claims = [f.claim for f in r.findings] if r else []
    return _claims_check("research_faithfulness", claims, _docs(r.sources) if r else "", config.MIN_FAITHFULNESS)


def brief_claims(b: Brief) -> list[str]:
    return [b.summary] + [f"{c.name}: {c.note}" for c in b.competitors] + list(b.key_observations)


def brief_groundedness(state: WorkflowState, brief: Optional[Brief] = None) -> MetricResult:
    b = brief or state.final or state.draft
    r = state.research
    evidence = "FINDINGS:\n" + "\n".join(f"- {f.claim}" for f in (r.findings if r else []))
    if r and r.sources:
        evidence += "\n\nDOCUMENTS:\n" + _docs(r.sources)
    return _claims_check("brief_groundedness", brief_claims(b) if b else [], evidence, config.MIN_GROUNDEDNESS)


def gap_honesty(state: WorkflowState) -> MetricResult:
    b = state.final or state.draft
    r = state.research
    res = _judge(Honesty,
                 "Decide whether the brief honestly admits missing information. honest=true if "
                 "missing_information lists what the evidence does not cover (e.g. market size, share, "
                 "growth) and the brief does not claim such facts. Return JSON {honest, reason}.",
                 f"Findings: {[f.claim for f in (r.findings if r else [])]}\nData gaps: {r.data_gaps if r else []}\n"
                 f"Brief missing_information: {b.missing_information if b else []}\nBrief summary: {b.summary if b else ''}")
    return MetricResult(name="gap_honesty", score=1.0 if res.honest else 0.0, passed=res.honest, reason=res.reason or "")
