"""Shared data structures passed between agent roles (Pydantic models)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    title: str
    url: str = ""
    snippet: str = Field(default="", description="Retrieved text (used by the evaluator).")


class Finding(BaseModel):
    claim: str = Field(description="One factual statement found during research.")
    source_title: str = Field(default="", description="Title of the source page.")
    source_url: str = Field(default="", description="URL of the source, empty for sample data.")


class ResearchResult(BaseModel):
    """What the Researcher is asked to produce."""
    findings: list[Finding] = Field(description="5-10 findings relevant to the query.")
    data_gaps: list[str] = Field(
        default_factory=list,
        description="Things that could not be found or verified.",
    )


class Research(ResearchResult):
    """Researcher output plus metadata added by code (not by the model)."""
    mode: Literal["web", "sample"] = "web"
    sources: list[Source] = Field(default_factory=list, description="Retrieved documents.")
    queries: list[str] = Field(default_factory=list, description="Search queries run (initial plan).")
    gap_rounds: list[list[str]] = Field(
        default_factory=list, description="Follow-up queries run to fill data gaps, one list per round."
    )


class Competitor(BaseModel):
    name: str
    note: str = Field(description="One sentence on why it matters, based on the findings.")


class Brief(BaseModel):
    summary: str = Field(description="2-4 sentence summary of the topic.")
    competitors: list[Competitor] = Field(
        default_factory=list, description="At most two competitors, if relevant."
    )
    key_observations: list[str] = Field(description="3-5 short observations.")
    sources: list[Source] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)

    @field_validator("competitors", mode="before")
    @classmethod
    def _max_two(cls, v):
        return v[:2] if isinstance(v, list) else v


IssueCategory = Literal["competitor_limit", "unsupported_claim", "missing_gap", "style", "other"]
BriefSection = Literal["summary", "competitors", "key_observations", "sources", "missing_information"]

# Lower number = fix first. Hard requirements beat evidence problems, which beat style.
ISSUE_PRIORITY: dict[str, int] = {
    "competitor_limit": 1, "unsupported_claim": 2, "missing_gap": 2, "other": 3, "style": 4,
}


class Issue(BaseModel):
    """One concrete problem the Critic found, so the Analyst knows exactly what to change."""
    section: BriefSection = Field(description="Part of the brief the problem is in.")
    category: IssueCategory = Field(
        description="competitor_limit: more than two competitors or an unsupported one. "
        "unsupported_claim: a statement the findings do not back up. "
        "missing_gap: a gap or caveat the brief fails to state. "
        "style: wording, vagueness or length. other: anything else."
    )
    comment: str = Field(description="What is wrong and what to do about it, in one or two sentences.")

    @property
    def priority(self) -> int:
        return ISSUE_PRIORITY.get(self.category, 3)


class Critique(BaseModel):
    approved: bool = Field(description="True if the brief needs no revision.")
    issues: list[Issue] = Field(default_factory=list)
    revision_instructions: str = Field(
        default="", description="Concrete instructions for the Analyst if not approved."
    )


class MetricResult(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str = ""
    details: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    metrics: list[MetricResult]
    verdict: Literal["RAG WORKING", "RAG NOT WORKING", "RAG NOT ACTIVE"]
    failed_checks: list[str] = Field(default_factory=list)


class StepMetric(BaseModel):
    """Cost of one workflow step: wall time and LLM usage."""
    step: str
    seconds: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class WorkflowState(BaseModel):
    """The single object handed from role to role."""
    query: str
    research: Optional[Research] = None
    draft: Optional[Brief] = None
    critique: Optional[Critique] = None
    final: Optional[Brief] = None
    revised: bool = False
    evaluation: Optional[EvaluationReport] = None
    metrics: list[StepMetric] = Field(default_factory=list)
    run_dir: Optional[str] = Field(default=None, description="Where run artifacts were saved, if enabled.")
    log: list[str] = Field(default_factory=list)
