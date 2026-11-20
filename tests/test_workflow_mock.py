"""End-to-end workflow + evaluator with the LLM and search mocked (no keys needed)."""
from market_research import workflow
from market_research.agents import analyst, critic, researcher  # noqa: F401  (ensure modules load)
from market_research.evaluation import judges, rules, tamper
from market_research.models import Brief, Competitor, Critique, Finding, ResearchResult, Source
from market_research.evaluation.judges import Claim as _Claim, Claims as _Claims, Honesty as _Honesty, Relevance as _Relevance
import importlib
from market_research import retrieval

# `market_research.agents` re-exports functions with the same names as its submodules,
# so fetch the submodules explicitly.
researcher_mod = importlib.import_module("market_research.agents.researcher")
analyst_mod = importlib.import_module("market_research.agents.analyst")
critic_mod = importlib.import_module("market_research.agents.critic")

SRCS = [Source(title=t, url=f"https://{t}.com/p", snippet=t * 60) for t in ("aa", "bb", "cc")]


def fake_ask_json(schema, system, user, model=None, temperature=0.3):
    if schema is ResearchResult:
        return ResearchResult(findings=[Finding(claim="f1", source_title="aa", source_url=SRCS[0].url)], data_gaps=["size"])
    if schema is Brief:
        return Brief(summary="s", competitors=[Competitor(name="X", note="n")], key_observations=["a", "b", "c"],
                     sources=[Source(title="aa", url=SRCS[0].url)], missing_information=["size"])
    if schema is Critique:
        return Critique(approved=True)
    if schema is _Relevance:
        return _Relevance(relevant_ids=[1, 2, 3])
    if schema is _Claims:
        n = user.count("\n") and len([l for l in user.split("Claims:")[1].splitlines() if l.strip()])
        # claims containing "340%" (the tampered one) are unsupported
        claims = [l.split(". ", 1)[1] for l in user.split("Claims:")[1].splitlines() if ". " in l]
        return _Claims(claims=[_Claim(claim=c, supported="340%" not in c) for c in claims])
    if schema is retrieval.QueryPlan:
        return retrieval.QueryPlan(queries=["q one", "q two"])
    if schema is retrieval.RelevantDocs:
        return retrieval.RelevantDocs(relevant_ids=[1, 2, 3])
    if schema is _Honesty:
        return _Honesty(honest=True, reason="ok")
    raise AssertionError(schema)


def setup(monkeypatch):
    for mod in (researcher_mod, analyst_mod, critic_mod, judges, retrieval):
        monkeypatch.setattr(mod, "ask_json", fake_ask_json)
    monkeypatch.setattr(researcher_mod, "web_search", lambda q, queries=None: SRCS)


def test_web_run_is_rag_working_and_tamper_is_caught(monkeypatch):
    setup(monkeypatch)
    st = workflow.orchestrator("q", evaluate_rag=True)
    assert st.research.mode == "web" and st.evaluation.verdict == "RAG WORKING"
    t = tamper(st.final)
    assert not rules.citation_validity(st, t).passed
    assert judges.brief_groundedness(st, t).score < judges.brief_groundedness(st).score


def test_sample_mode_is_not_active(monkeypatch):
    setup(monkeypatch)
    st = workflow.orchestrator("q", use_sample=True, evaluate_rag=True)
    assert st.evaluation.verdict == "RAG NOT ACTIVE"


def test_forced_revision_happens_exactly_once(monkeypatch):
    setup(monkeypatch)
    st = workflow.orchestrator("q", force_revision=True)
    assert st.revised and sum("revising" in l for l in st.log) == 1


def test_irrelevant_documents_are_filtered_out(monkeypatch):
    setup(monkeypatch)
    monkeypatch.setattr(retrieval, "ask_json", lambda schema, *a, **k: (
        retrieval.RelevantDocs(relevant_ids=[1]) if schema is retrieval.RelevantDocs
        else retrieval.QueryPlan(queries=["q"])))
    kept = retrieval.select_relevant("topic", SRCS)
    assert [s.title for s in kept] == ["aa"]


def test_graph_revise_branch_and_ragas_is_optional(monkeypatch):
    from market_research.evaluation import verdict
    setup(monkeypatch)
    calls = []
    monkeypatch.setattr("market_research.evaluation.ragas_eval.ragas_metrics",
                        lambda s: calls.append(1) or [])
    s = workflow.orchestrator("x", evaluate_rag=True, force_revision=True)
    assert s.revised and s.evaluation.verdict == "RAG WORKING" and not calls  # ragas off by default
    s = workflow.orchestrator("x", evaluate_rag=True, ragas=True)
    assert not s.revised and calls == [1] and s.evaluation.verdict == "RAG WORKING"


def test_ragas_failure_is_non_blocking(monkeypatch):
    from market_research.evaluation import ragas_eval
    setup(monkeypatch)
    monkeypatch.setattr("ragas.evaluate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("market_research.llm.chat_model", lambda *a, **k: object())
    s = workflow.orchestrator("x", evaluate_rag=True, ragas=True)
    names = {m.name: m for m in s.evaluation.metrics}
    assert names["ragas_faithfulness"].reason.startswith("skipped")
    assert s.evaluation.verdict == "RAG WORKING"
