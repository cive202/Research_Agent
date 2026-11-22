"""Shared data structures passed between agent roles (Pydantic models)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .guideline import Guideline


class Source(BaseModel):
    title: str
    url: str = ""
    snippet: str = Field(default="", description="Retrieved text (used by the evaluator).")
    on_focus: bool = Field(default=True, description="Set by code: False if it does not mention the focus.")


MAX_COMPETITORS = 2


class FindingDraft(BaseModel):
    """What the model writes for one finding. The ID and snippet are added by code, never by the model."""
    claim: str = Field(description="One factual statement found during research.")
    source_title: str = Field(default="", description="Title of the source page.")
    source_url: str = Field(default="", description="URL of the source, empty for sample data.")


class Finding(FindingDraft):
    id: str = Field(default="", description="F1, F2, ... assigned by code after extraction.")
    snippet: str = Field(default="", description="Short excerpt of the source text, set by code.")
    on_focus: bool = Field(default=True, description="Set by code. False = BACKGROUND: does not mention the focus.")


class ResearchResult(BaseModel):
    """What the Researcher is asked to produce."""
    findings: list[FindingDraft] = Field(description="5-10 findings relevant to the query.")
    data_gaps: list[str] = Field(
        default_factory=list,
        description="Things that could not be found or verified.",
    )


class Research(ResearchResult):
    """Researcher output plus metadata added by code (not by the model)."""
    findings: list[Finding] = Field(default_factory=list, description="Numbered findings F1, F2, ...")
    mode: Literal["web", "sample"] = "web"
    provider: str = Field(default="", description="Search provider used: tavily, duckduckgo or sample.")
    focus: list[str] = Field(default_factory=list, description="Place(s) or company the research is about.")
    sources: list[Source] = Field(default_factory=list, description="Retrieved documents.")
    queries: list[str] = Field(default_factory=list, description="Search queries run (initial plan).")
    gap_rounds: list[list[str]] = Field(
        default_factory=list, description="Follow-up queries run to fill data gaps, one list per round."
    )


class Competitor(BaseModel):
    name: str
    note: str = Field(description="One sentence on why it matters. Must end with finding citations like [F2].")


class Brief(BaseModel):
    summary: str = Field(description="2-4 sentence summary of the topic.")
    competitors: list[Competitor] = Field(
        default_factory=list, description="At most two competitors, if relevant."
    )
    key_observations: list[str] = Field(
        description="3-5 short observations. Each must cite finding IDs, for example 'Foodmandu leads Kathmandu [F2].'")
    sources: list[Source] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    competitors_dropped: int = Field(default=0, description="Set by code. Always return 0.")

    @model_validator(mode="before")
    @classmethod
    def _cap_competitors(cls, data):
        """Hard limit in code: keep at most two competitors and remember how many were cut."""
        comps = data.get("competitors") if isinstance(data, dict) else None
        if isinstance(comps, list) and len(comps) > MAX_COMPETITORS:
            data = {**data, "competitors": comps[:MAX_COMPETITORS],
                    "competitors_dropped": len(comps) - MAX_COMPETITORS}
        return data


IssueCategory = Literal["competitor_limit", "unsupported_claim", "missing_gap", "style", "other"]
BriefSection = Literal["summary", "competitors", "key_observations", "sources", "missing_information"]

# Which rule an issue breaks. Reviews clash, so the editor resolves them in this order:
# the human guideline first, then the research (no hallucinations), then the writing profiles.
IssueRule = Literal["guideline", "research", "structure", "terminology", "character"]
RULE_PRIORITY: dict[str, int] = {"guideline": 1, "research": 2, "structure": 3, "terminology": 4, "character": 5}

# Within a rule: hard requirements beat evidence problems, which beat style.
ISSUE_PRIORITY: dict[str, int] = {
    "competitor_limit": 1, "unsupported_claim": 2, "missing_gap": 2, "other": 3, "style": 4,
}


class Issue(BaseModel):
    """One concrete problem the Critic found, so the Analyst knows exactly what to change."""
    rule: IssueRule = Field(
        default="research",
        description="Which rule is broken. guideline: the brief ignores the guideline's angle, audience, "
        "key questions, must-include items or constraints. research: a claim the findings do not support, "
        "or a gap the brief hides. structure / terminology / character: the brief breaks that writing profile.",
    )
    section: BriefSection = Field(description="Part of the brief the problem is in.")
    category: IssueCategory = Field(
        description="competitor_limit: more than two competitors or an unsupported one. "
        "unsupported_claim: a statement the findings do not back up. "
        "missing_gap: a gap or caveat the brief fails to state. "
        "style: wording, vagueness or length. other: anything else."
    )
    comment: str = Field(description="What is wrong and what to do about it, in one or two sentences.")

    @property
    def priority(self) -> tuple[int, int]:
        return RULE_PRIORITY.get(self.rule, 3), ISSUE_PRIORITY.get(self.category, 3)


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


GroundingCode = Literal["uncited_observation", "unknown_citation", "too_many_competitors",
                        "unsupported_competitor", "background_competitor", "unknown_source"]


class GroundingFlag(BaseModel):
    """One evidence problem found by the plain-Python grounding check (no LLM)."""
    code: GroundingCode
    section: str
    message: str


class TraceStep(BaseModel):
    """One step of the run: which role ran, how it ended and how long it took."""
    role: str
    status: Literal["ok", "error"] = "ok"
    seconds: float = 0.0
    detail: str = ""


class WorkflowState(BaseModel):
    """The single object handed from role to role."""
    query: str
    guideline: Optional[Guideline] = Field(default=None, description="Human-written guideline, if one was given.")
    research: Optional[Research] = None
    draft: Optional[Brief] = Field(default=None, description="First draft (v0).")
    critique: Optional[Critique] = Field(default=None, description="Latest review.")
    final: Optional[Brief] = None
    revised: bool = False
    drafts: list[Brief] = Field(default_factory=list, description="Every version, v0 first.")
    critiques: list[Critique] = Field(default_factory=list, description="One review per round; critiques[i] reviews drafts[i].")
    max_iterations: int = 1
    evaluation: Optional[EvaluationReport] = None
    metrics: list[StepMetric] = Field(default_factory=list)
    revision_error: str = Field(default="", description="Why the revision failed, if it did (the draft was kept).")
    run_dir: Optional[str] = Field(default=None, description="Where run artifacts were saved, if enabled.")
    log: list[str] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list, description="Per-run agent trace.")
    grounding: list[GroundingFlag] = Field(default_factory=list, description="Flags for the final brief.")
    error: Optional[str] = Field(default=None, description="Friendly message if the run stopped early.")
