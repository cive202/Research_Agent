"""Evidence grounding, hard limits and sample-data labelling. LLM and search are mocked: no API calls."""
import pytest

from market_research import config, search, workflow
from market_research.agents.analyst import NO_FIGURES, finalize_brief
from market_research.grounding import check_grounding, cited_ids, derive_sources
from market_research.llm import explain_error
from market_research.models import Brief, Competitor, Critique, Finding, Research, Source
from market_research.rendering import SAMPLE_WARNING, brief_to_markdown, state_to_json
from market_research.search import SearchUnavailable
from tests.test_workflow_mock import analyst_mod, critic_mod, fake_ask_json, researcher_mod, setup

URL1, URL2 = "https://aa.com/p", "https://bb.com/p"


def research(n: int = 3, mode: str = "web") -> Research:
    findings = [Finding(id=f"F{i}", claim=f"claim {i}", source_title=f"t{i}", source_url=f"https://s{i}.com/p")
                for i in range(1, n + 1)]
    return Research(mode=mode, findings=findings, sources=[Source(title=f"t{i}", url=f.source_url, snippet="x")
                                                           for i, f in enumerate(findings, 1)])


def brief(**kw) -> Brief:
    base = dict(summary="s", key_observations=["a [F1]", "b [F2]", "c [F3]"],
                competitors=[Competitor(name="X", note="n [F1]")])
    base.update(kw)
    return Brief(**base)


def codes(b: Brief, r: Research) -> list[str]:
    return [f.code for f in check_grounding(b, r)]


def test_cited_ids_reads_single_multiple_and_comma_forms():
    assert cited_ids("x [F2][F1] and [F3, F4]") == ["F2", "F1", "F3", "F4"]
    assert cited_ids("no citation (F1)") == []


def test_clean_brief_has_no_flags():
    assert codes(brief(), research()) == []


def test_uncited_observation_is_flagged():
    assert codes(brief(key_observations=["a [F1]", "no citation here", "c [F3]"]), research()) == ["uncited_observation"]


def test_citation_to_missing_finding_is_flagged():
    assert "unknown_citation" in codes(brief(key_observations=["a [F9]", "b [F2]", "c [F3]"]), research())


def test_competitor_without_supporting_finding_is_flagged():
    assert codes(brief(competitors=[Competitor(name="X", note="no cite")]), research()) == ["unsupported_competitor"]


def test_source_url_not_in_findings_is_flagged():
    b = brief(sources=[Source(title="made up", url="https://invented.example/x")])
    assert codes(b, research()) == ["unknown_source"]


def test_more_than_two_competitors_is_cut_in_code_and_flagged():
    comps = [Competitor(name=str(i), note="n [F1]") for i in range(4)]
    b = brief(competitors=comps)
    assert len(b.competitors) == 2 and b.competitors_dropped == 2  # enforced by the model, not the prompt
    assert codes(b, research()) == ["too_many_competitors"]


def test_sources_are_derived_from_cited_findings_only():
    r = research(4)
    b = brief(key_observations=["a [F2]", "b [F2]", "c [F4]"], competitors=[], sources=[Source(title="junk", url="x")])
    out = finalize_brief(b, r)
    assert [s.url for s in out.sources] == [r.findings[1].source_url, r.findings[3].source_url]
    assert [s.url for s in derive_sources(b, r)] == [s.url for s in out.sources]


def test_thin_evidence_and_missing_figures_are_stated_honestly():
    out = finalize_brief(brief(), research(2))
    text = " ".join(out.missing_information)
    assert "thin" in text and NO_FIGURES in out.missing_information
    rich = research(config.THIN_EVIDENCE_FINDINGS)
    rich.findings[0].claim = "The market is worth $120 million, growing 15%."
    assert finalize_brief(brief(), rich).missing_information == []


def test_critic_approval_is_overruled_when_grounding_flags_exist(monkeypatch):
    setup(monkeypatch)
    uncited = Brief(summary="s", key_observations=["a", "b", "c"], competitors=[Competitor(name="X", note="n [F1]")])
    calls = {"edit": 0}

    def ask(schema, system, user, model=None, temperature=0.3):
        if schema is Brief:
            if "Critic issues" in user:
                calls["edit"] += 1
                return brief(key_observations=["a [F1]", "b [F1]", "c [F1]"])
            return uncited.model_copy(deep=True)
        return Critique(approved=True) if schema is Critique else fake_ask_json(schema, system, user)

    for mod in (analyst_mod, critic_mod):
        monkeypatch.setattr(mod, "ask_json", ask)
    st = workflow.orchestrator("q")
    assert calls["edit"] == 1 and st.revised and not st.critiques[0].approved
    assert st.grounding == []  # the revised brief is grounded


def test_critic_prompt_contains_the_grounding_flags(monkeypatch):
    setup(monkeypatch)
    seen = {}

    def ask(schema, system, user, model=None, temperature=0.3):
        if schema is Critique:
            seen["user"] = user
        if schema is Brief:
            return Brief(summary="s", key_observations=["uncited one", "b [F1]", "c [F1]"])
        return Critique(approved=True) if schema is Critique else fake_ask_json(schema, system, user)

    for mod in (analyst_mod, critic_mod):
        monkeypatch.setattr(mod, "ask_json", ask)
    workflow.orchestrator("q")
    assert "uncited_observation" in seen["user"]


def test_critic_rejecting_forever_still_gives_one_revision(monkeypatch):
    setup(monkeypatch)
    n = {"critic": 0, "edit": 0}

    def ask(schema, system, user, model=None, temperature=0.3):
        if schema is Critique:
            n["critic"] += 1
            return Critique(approved=False, revision_instructions="again")
        if schema is Brief and "Critic issues" in user:
            n["edit"] += 1
        return fake_ask_json(schema, system, user)

    for mod in (analyst_mod, critic_mod):
        monkeypatch.setattr(mod, "ask_json", ask)
    st = workflow.orchestrator("q", iterations=99)
    assert n == {"critic": 1, "edit": 1} and st.max_iterations == 1 and len(st.drafts) == 2


def test_search_failure_falls_back_to_clearly_labelled_sample_data(monkeypatch):
    setup(monkeypatch)

    def boom(q, queries=None):
        raise SearchUnavailable("no network")

    monkeypatch.setattr(researcher_mod, "web_search", boom)
    st = workflow.orchestrator("Analyze the food delivery market in Nepal")
    assert st.research.mode == "sample" and st.research.provider == "sample"
    assert all(f.claim.startswith("SAMPLE") for f in st.research.findings)
    assert any("unavailable" in g for g in st.research.data_gaps)
    assert st.final.summary.startswith("SAMPLE DATA")
    assert SAMPLE_WARNING in brief_to_markdown(st)
    assert '"data_mode": "sample"' in state_to_json(st) and "SAMPLE DATA MODE" in state_to_json(st)
    assert any("SAMPLE" in s.title for s in st.final.sources)


def test_handoff_order_and_trace(monkeypatch):
    setup(monkeypatch)
    st = workflow.orchestrator("q")
    assert [t.role for t in st.trace] == ["Orchestrator", "Researcher", "Analyst", "Critic", "Orchestrator", "Orchestrator"]
    assert all(t.status == "ok" for t in st.trace)
    roles = [m.split(":")[0] for m in st.log]
    assert roles.index("Researcher") < roles.index("Analyst") < roles.index("Critic")
    assert st.research and st.draft and st.critique and st.final  # data handed over as structured state


def test_failure_is_friendly_and_keeps_the_partial_state(monkeypatch):
    setup(monkeypatch)

    def ask(schema, system, user, model=None, temperature=0.3):
        if schema is Brief:
            raise ConnectionError("connection refused by host")
        return fake_ask_json(schema, system, user)

    monkeypatch.setattr(analyst_mod, "ask_json", ask)
    st = workflow.orchestrator("q", raise_errors=False)
    assert "internet connection" in st.error and "Traceback" not in st.error
    assert st.research is not None and st.final is None
    assert [t.status for t in st.trace][-1] == "error"
    with pytest.raises(ConnectionError):
        workflow.orchestrator("q")


@pytest.mark.parametrize("exc,word", [
    (RuntimeError("Error code: 429 rate limit"), "rate limit"),
    (RuntimeError("401 invalid api key"), "GROQ_API_KEY"),
    (RuntimeError("Connection error."), "internet"),
    (ConnectionError("refused"), "internet"),
    (RuntimeError("weird"), "retry"),
])
def test_explain_error_maps_to_friendly_messages(exc, word):
    assert word in explain_error(exc)


def test_error_class_ignores_words_echoed_from_model_output():
    class BadRequest(Exception):
        status_code = 400

    msg = "Error code: 400 - " + "x" * 100 + " the network and connection and timeout of delivery"
    assert "internet" not in explain_error(BadRequest(msg))


def test_groq_json_validate_failed_is_retried_once(monkeypatch):
    from pydantic import BaseModel

    from market_research import llm

    class Bad(Exception):
        status_code = 400

    class Out(BaseModel):
        x: int

    from langchain_core.runnables import RunnableLambda

    calls = []

    def fake_model(_prompt_value):
        calls.append(1)
        if len(calls) == 1:
            raise Bad("json_validate_failed: Failed to generate JSON")
        return Out(x=1)

    class Model:
        def with_structured_output(self, *a, **k):
            return RunnableLambda(fake_model)

    monkeypatch.setattr(llm, "chat_model", lambda *a, **k: Model())
    assert llm.ask_json(Out, "s", "u").x == 1 and len(calls) == 2


def test_search_provider_order_tavily_then_duckduckgo(monkeypatch):
    rows = [{"title": "t", "url": "https://x.com/a", "content": "c" * 200}]

    def make(name, fail=False):
        def fetch(text):
            if fail:
                raise RuntimeError("down")
            return rows
        return fetch

    monkeypatch.setattr(config, "tavily_api_key", lambda: "key")
    monkeypatch.setattr(search, "_tavily_fetch", lambda key: make("tavily"))
    monkeypatch.setattr(search, "_ddgs_fetch", lambda: make("ddg"))
    assert search.web_search("q") and search.last_provider() == "tavily"

    monkeypatch.setattr(search, "_tavily_fetch", lambda key: make("tavily", fail=True))
    assert search.web_search("q") and search.last_provider() == "duckduckgo"

    monkeypatch.setattr(config, "tavily_api_key", lambda: None)
    monkeypatch.setattr(search, "_ddgs_fetch", lambda: make("ddg", fail=True))
    with pytest.raises(SearchUnavailable):
        search.web_search("q")


def test_findings_are_numbered_capped_and_deduplicated_by_url():
    from market_research.agents.researcher import number_findings

    srcs = [Source(title="a", url="https://a.com/x/", snippet="snippet a")]
    many = [Finding(claim=f"c{i}", source_url="https://www.a.com/x") for i in range(15)]
    out = number_findings(many, srcs)
    assert [f.id for f in out] == [f"F{i}" for i in range(1, config.MAX_FINDINGS + 1)]
    assert out[0].snippet == "snippet a"


def test_intake_declines_with_a_clarifying_question_and_fails_open(monkeypatch):
    from market_research import intake
    from market_research.llm import LLMError

    monkeypatch.setenv("INTAKE_CHECK", "1")
    setup(monkeypatch)
    answer = {"v": intake.Intake(is_valid=False, clarifying_question="Which market do you mean?")}
    monkeypatch.setattr(intake, "ask_json", lambda *a, **k: answer["v"])
    with pytest.raises(ValueError, match="Which market"):
        workflow.orchestrator("market")

    answer["v"] = intake.Intake(is_valid=True)
    assert workflow.orchestrator("food delivery in Nepal").final

    def down(*a, **k):
        raise LLMError("down")

    monkeypatch.setattr(intake, "ask_json", down)
    assert workflow.orchestrator("food delivery in Nepal").final  # fails open
