"""Focus enforcement, today's date in search planning, prompt-injection guard and the gap-rounds option."""
from datetime import date

from market_research import config, retrieval, workflow
from market_research.focus import clean_focus, ensure_focus, mentions_focus, rank_by_focus
from market_research.grounding import check_grounding
from market_research.models import Brief, Competitor, Finding, Research, ResearchResult, Source
from tests.test_improvements import gap_setup
from tests.test_workflow_mock import analyst_mod, critic_mod, fake_ask_json, researcher_mod, setup

NEPAL = [Source(title=f"Nepal {t}", url=f"https://{t}.com.np/nepal-food", snippet=f"Kathmandu {t} " * 20)
         for t in ("aa", "bb", "cc")]
GLOBAL = [Source(title=f"Global report {n}", url=f"https://g{n}.com/report", snippet="worldwide delivery " * 20)
          for n in range(5)]


def test_focus_helpers():
    assert clean_focus([" Nepal ", "nepal", "global", "market", "Kathmandu"]) == ["Nepal", "Kathmandu"]
    assert mentions_focus(["Food in NEPAL today"], ["Nepal"])
    assert not mentions_focus(["Nepalese cuisine"], ["Nepal"])  # whole words only
    assert mentions_focus(["anything"], [])  # no focus: nothing to enforce
    assert ensure_focus(["food delivery Nepal", "Foodmandu funding"], ["Nepal"]) == [
        "food delivery Nepal", "Foodmandu funding Nepal"]


def test_rank_puts_focus_first_and_caps_background():
    ranked = rank_by_focus(GLOBAL[:2] + NEPAL + GLOBAL[2:], ["Nepal"], max_background=3)
    assert [s.on_focus for s in ranked] == [True] * 3 + [False] * 3
    assert ranked[3].title == "Global report 0"  # background keeps search order


def test_plan_search_adds_focus_date_and_recency(monkeypatch):
    seen = {}

    def ask(schema, system, user, **k):
        seen.update(system=system, user=user)
        return retrieval.QueryPlan(queries=["food delivery market size", "Foodmandu Nepal funding"],
                                   focus=["Nepal", "market", "Kathmandu"])

    monkeypatch.setattr(retrieval, "ask_json", ask)
    plan = retrieval.plan_search("Analyze the food delivery market in Nepal")
    assert plan.focus == ["Nepal", "Kathmandu"]
    assert plan.queries == ["food delivery market size Nepal", "Foodmandu Nepal funding"]
    assert f"Today's date: {date.today():%Y-%m-%d}" in seen["user"]
    assert "do not put a past year" in seen["system"]

    retrieval.plan_gap_queries("topic", ["market size"], [], ["Nepal"])
    assert f"Today's date: {date.today():%Y-%m-%d}" in seen["user"] and "past year" in seen["system"]


def test_researcher_labels_background_sources_and_findings(monkeypatch):
    setup(monkeypatch)
    prompts = []

    def ask(schema, system, user, model=None, temperature=0.3):
        if schema is retrieval.QueryPlan:
            return retrieval.QueryPlan(queries=["q"], focus=["Nepal"])
        if schema is retrieval.RelevantDocs:
            return retrieval.RelevantDocs(relevant_ids=list(range(1, 9)))
        if schema is ResearchResult:
            prompts.append(user)
            return ResearchResult(findings=[
                Finding(claim="Foodmandu leads", source_url=NEPAL[0].url),
                Finding(claim="Global market is $100bn", source_url=GLOBAL[0].url),
                Finding(claim="Nepal's riders grew", source_url=GLOBAL[1].url),  # claim names the focus
            ], data_gaps=[])
        return fake_ask_json(schema, system, user, model, temperature)

    for mod in (researcher_mod, retrieval):
        monkeypatch.setattr(mod, "ask_json", ask)
    monkeypatch.setattr(researcher_mod, "web_search", lambda q, queries=None: GLOBAL + NEPAL)
    st = workflow.orchestrator("food delivery in Nepal")
    r = st.research
    assert r.focus == ["Nepal"]
    assert [s.on_focus for s in r.sources] == [True] * 3 + [False] * config.MAX_BACKGROUND_SOURCES
    assert [f.on_focus for f in r.findings] == [True, False, True]
    assert "(BACKGROUND) Global report 0" in prompts[0] and "Focus: Nepal" in prompts[0]
    assert any("focus Nepal" in line for line in st.log)


def test_competitor_backed_only_by_background_is_flagged():
    research = Research(focus=["Nepal"], findings=[
        Finding(id="F1", claim="Foodmandu leads", on_focus=True),
        Finding(id="F2", claim="DoorDash is the global leader", on_focus=False)])
    brief = Brief(summary="s", key_observations=["a [F1]", "b [F1]", "c [F1]"],
                  competitors=[Competitor(name="Foodmandu", note="leads [F1]"),
                               Competitor(name="DoorDash", note="global leader [F2]")])
    codes = [(f.code, f.message) for f in check_grounding(brief, research)]
    assert [c for c, _ in codes] == ["background_competitor"] and "DoorDash" in codes[0][1]
    research.focus = []  # without a focus there is no background, so nothing to flag
    assert check_grounding(brief, research) == []


def test_every_prompt_with_web_text_has_the_injection_guard(monkeypatch):
    from market_research.evaluation import judges
    systems = []
    setup(monkeypatch)

    def ask(schema, system, user, model=None, temperature=0.3):
        systems.append((schema.__name__, system))
        return fake_ask_json(schema, system, user, model, temperature)

    for mod in (researcher_mod, analyst_mod, critic_mod, retrieval, judges):
        monkeypatch.setattr(mod, "ask_json", ask)
    workflow.orchestrator("q", evaluate_rag=True)
    web_text = {"ResearchResult", "RelevantDocs", "Brief", "Critique", "Claims", "Relevance", "Honesty"}
    checked = {name for name, system in systems if name in web_text}
    assert checked == web_text
    assert all(config.UNTRUSTED_NOTE in system for name, system in systems if name in web_text)


def test_gap_rounds_option_overrides_config(monkeypatch):
    searches, extractions = [], []
    gap_setup(monkeypatch, searches, extractions)
    off = workflow.orchestrator("q", max_gap_rounds=0)
    assert off.research.gap_rounds == [] and len(extractions) == 1
    extractions.clear()
    monkeypatch.setattr(config, "MAX_GAP_ROUNDS", 0)  # config says off; the explicit option wins
    on = workflow.orchestrator("q", max_gap_rounds=1)
    assert len(on.research.gap_rounds) == 1
