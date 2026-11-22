"""Gap-driven research, typed critique, style check and run artifacts (LLM and search mocked)."""
import json

from market_research import config, retrieval, workflow
from market_research.evaluation.rules import style_check
from market_research.models import Brief, Competitor, Critique, Finding, Issue, ResearchResult, Source, WorkflowState
from tests.test_workflow_mock import SRCS, analyst_mod, fake_ask_json, researcher_mod, setup

NEW = [Source(title="dd", url="https://dd.com/p", snippet="dd" * 80)]


def gap_setup(monkeypatch, searches, extractions):
    """Researcher whose first extraction reports a gap and whose follow-up search finds a new source."""
    setup(monkeypatch)

    def ask(schema, system, user, model=None, temperature=0.3):
        if schema is ResearchResult:
            extractions.append(user.count("\n["))
            gaps = ["market size"] if len(extractions) == 1 else []
            return ResearchResult(findings=[Finding(claim="f", source_url=SRCS[0].url)], data_gaps=gaps)
        if schema is retrieval.QueryPlan:
            return retrieval.QueryPlan(queries=["gap query about market size"] if "Data gaps" in user else ["initial q"])
        if schema is retrieval.RelevantDocs:
            return retrieval.RelevantDocs(relevant_ids=[1, 2, 3])
        return fake_ask_json(schema, system, user, model, temperature)

    def search(q, queries=None):
        searches.append(queries)
        return SRCS + NEW if queries and "gap query about market size" in queries else SRCS

    for mod in (researcher_mod, retrieval):
        monkeypatch.setattr(mod, "ask_json", ask)
    monkeypatch.setattr(researcher_mod, "web_search", search)


def test_gap_round_searches_for_missing_facts_and_adds_sources(monkeypatch):
    searches, extractions = [], []
    gap_setup(monkeypatch, searches, extractions)
    st = workflow.orchestrator("q")
    assert searches[-1] == ["gap query about market size"]
    assert st.research.gap_rounds == [["gap query about market size"]]
    assert "https://dd.com/p" in {s.url for s in st.research.sources}
    assert len(extractions) == 2 and st.research.data_gaps == []
    assert any("gap round 1" in line for line in st.log)


def test_gap_loop_stops_when_nothing_new_is_found(monkeypatch):
    searches, extractions = [], []
    gap_setup(monkeypatch, searches, extractions)
    monkeypatch.setattr(researcher_mod, "web_search", lambda q, queries=None: SRCS)  # never anything new
    st = workflow.orchestrator("q")
    assert len(extractions) == 1  # no pointless re-extraction
    assert len(st.research.gap_rounds) == 1


def test_gap_loop_can_be_disabled(monkeypatch):
    searches, extractions = [], []
    gap_setup(monkeypatch, searches, extractions)
    monkeypatch.setattr(config, "MAX_GAP_ROUNDS", 0)
    st = workflow.orchestrator("q")
    assert st.research.gap_rounds == [] and len(extractions) == 1


def test_issues_are_prioritised_and_reach_the_analyst(monkeypatch):
    setup(monkeypatch)
    issues = [Issue(section="summary", category="style", comment="vague"),
              Issue(section="competitors", category="competitor_limit", comment="three listed"),
              Issue(section="key_observations", category="unsupported_claim", comment="no source")]
    seen = {}

    def ask(schema, system, user, model=None, temperature=0.3):
        if schema is Critique:
            return Critique(approved=False, issues=list(issues), revision_instructions="fix")
        if schema is Brief and "Critic issues" in user:
            seen["prompt"] = user
        return fake_ask_json(schema, system, user, model, temperature)

    from tests.test_workflow_mock import critic_mod
    for mod in (analyst_mod, critic_mod):
        monkeypatch.setattr(mod, "ask_json", ask)
    st = workflow.orchestrator("q")
    assert [i.category for i in st.critique.issues] == ["competitor_limit", "unsupported_claim", "style"]
    p = seen["prompt"]
    assert p.index("competitor_limit") < p.index("unsupported_claim") < p.index("/style]")
    assert st.revised and st.draft is not st.final  # first draft kept next to the revision


def test_style_check_flags_slop_and_vague_claims():
    clean = Brief(summary="Foodmandu leads with 40% share.", key_observations=["a", "b", "c"])
    sloppy = Brief(summary="A rapidly evolving market — most companies delve into it.",
                   competitors=[Competitor(name="X", note="n")], key_observations=["a", "b", "c"])
    st = WorkflowState(query="q")
    assert style_check(st, clean).passed
    r = style_check(st, sloppy)
    assert not r.passed and len(r.details) == 4  # rapidly evolving, delve, most companies, em dash


def test_step_metrics_and_run_artifacts(monkeypatch, tmp_path):
    setup(monkeypatch)
    st = workflow.orchestrator("Nepal food delivery!", evaluate_rag=True, save_dir=str(tmp_path))
    assert [m.step for m in st.metrics][:3] == ["researcher", "analyst", "critic"]
    assert all(m.seconds >= 0 for m in st.metrics)
    from pathlib import Path
    run = Path(st.run_dir)
    assert run.parent == tmp_path and "nepal-food-delivery" in run.name
    assert {p.name for p in run.iterdir()} == {"research.md", "draft_v0.md", "reviews.md", "final.md", "state.json"}
    assert json.loads((run / "state.json").read_text())["query"] == "Nepal food delivery!"
    assert "## Findings" in (run / "research.md").read_text()


def test_failed_gap_extraction_keeps_earlier_findings(monkeypatch):
    from market_research.llm import LLMError
    searches, extractions = [], []
    gap_setup(monkeypatch, searches, extractions)
    inner = researcher_mod.ask_json

    def flaky(schema, system, user, model=None, temperature=0.3):
        if schema is ResearchResult and extractions:  # second extraction fails (e.g. rate limit)
            raise LLMError("rate limit")
        return inner(schema, system, user, model, temperature)

    monkeypatch.setattr(researcher_mod, "ask_json", flaky)
    st = workflow.orchestrator("q")
    assert len(st.research.findings) == 1 and st.research.data_gaps == ["market size"]
    assert {s.url for s in st.research.sources} == {s.url for s in SRCS}  # new sources not half-applied
