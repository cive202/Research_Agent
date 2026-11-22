"""Guideline input, profiles, review loop, file-based MCP tools and judge calibration (LLM and search mocked)."""
import json

import pytest

from market_research import config, tools, workflow
from market_research.evaluation import calibration
from market_research.guideline import parse_guideline
from market_research.models import Brief, Critique, Issue, Research
from market_research.profiles import load_examples, load_profiles, writing_context
from tests.test_workflow_mock import SRCS, analyst_mod, critic_mod, fake_ask_json, researcher_mod, setup

GUIDE = """# Food delivery in Nepal
<!-- comment -->
## Angle
Is there room for a third app?
## Audience
Investor.
## Key questions
- Foodmandu market share
- Nepal food delivery market size
  in 2024
-
## Seed links
- https://example.com/a and (https://example.com/b)
## Constraints
- Summary at most 2 sentences.
"""


def test_parse_guideline_handles_template_noise():
    g = parse_guideline(GUIDE)
    assert g.topic == "Food delivery in Nepal" and g.angle == "Is there room for a third app?"
    assert g.key_questions == ["Foodmandu market share", "Nepal food delivery market size in 2024"]
    assert g.seed_links == ["https://example.com/a", "https://example.com/b"]
    assert "Constraints" in g.to_prompt() and "Audience: Investor." in g.to_prompt()
    with pytest.raises(ValueError):
        parse_guideline("## Angle\nno title")


def test_profiles_and_examples_load_from_markdown():
    p = load_profiles()
    assert set(p) == {"structure", "terminology", "character"}
    assert "delve" in p["terminology"]  # banned list appended from config
    assert len(load_examples()) == config.MAX_FEW_SHOT
    assert "fictional" in writing_context().lower()


def test_guideline_drives_queries_and_reaches_writer_and_critic(monkeypatch):
    setup(monkeypatch)
    searched, prompts = [], {}
    monkeypatch.setattr(researcher_mod, "web_search", lambda q, queries=None: searched.append(queries) or SRCS)
    monkeypatch.setattr(researcher_mod, "read_urls", lambda urls: [])

    def ask(schema, system, user, model=None, temperature=0.3):
        prompts.setdefault(schema.__name__, user)
        return fake_ask_json(schema, system, user, model, temperature)

    for mod in (analyst_mod, critic_mod):
        monkeypatch.setattr(mod, "ask_json", ask)
    st = workflow.orchestrator("ignored", guideline=parse_guideline(GUIDE))
    assert st.query == "Food delivery in Nepal"
    assert searched[0][:2] == ["Foodmandu market share", "Nepal food delivery market size in 2024"]
    for name in ("Brief", "Critique"):
        assert "Is there room for a third app?" in prompts[name] and "STRUCTURE PROFILE" in prompts[name]
    assert "EXAMPLE BRIEF" in prompts["Brief"] and "EXAMPLE BRIEF" not in prompts["Critique"]


def test_critic_rejecting_every_round_still_gives_only_one_revision(monkeypatch, tmp_path):
    setup(monkeypatch)
    calls = {"critic": 0, "edit": 0}

    def ask(schema, system, user, model=None, temperature=0.3):
        if schema is Critique:
            calls["critic"] += 1
            return Critique(approved=False, issues=[
                Issue(rule="terminology", section="summary", category="style", comment="banned word"),
                Issue(rule="guideline", section="summary", category="other", comment="angle not answered"),
                Issue(rule="research", section="key_observations", category="unsupported_claim", comment="no source"),
            ])
        if schema is Brief and "Critic issues" in user:
            calls["edit"] += 1
            # guideline beats research beats profiles
            assert user.index("[guideline/") < user.index("[research/") < user.index("[terminology/")
            b = fake_ask_json(schema, system, user)
            b.summary = f"v{calls['edit']}"
            return b
        return fake_ask_json(schema, system, user, model, temperature)

    for mod in (analyst_mod, critic_mod):
        monkeypatch.setattr(mod, "ask_json", ask)
    st = workflow.orchestrator("q", iterations=3, save_dir=str(tmp_path))  # 3 is clamped to the hard cap of 1
    assert calls == {"critic": 1, "edit": 1}
    assert [d.summary for d in st.drafts] == ["s", "v1"] and st.final.summary == "v1"
    assert len(st.critiques) == 1 and st.draft.summary == "s" and st.revised
    files = {p.name for p in (tmp_path / st.run_dir.split("/")[-1]).iterdir()}
    assert {"draft_v0.md", "draft_v1.md", "reviews.md"} <= files


def test_review_loop_stops_when_approved(monkeypatch):
    setup(monkeypatch)
    st = workflow.orchestrator("q", iterations=4)
    assert len(st.drafts) == 1 and len(st.critiques) == 1 and not st.revised


def test_mcp_tools_end_to_end(monkeypatch, tmp_path):
    setup(monkeypatch)
    import market_research.retrieval as retrieval
    import market_research.search as search

    monkeypatch.setattr(search, "web_search", lambda q, queries=None: SRCS)
    monkeypatch.setattr(search, "read_urls", lambda urls: SRCS[:1])
    monkeypatch.setattr(retrieval, "ask_json", fake_ask_json)
    (tmp_path / "guideline.md").write_text(GUIDE)

    r = tools.read_links(str(tmp_path))
    assert r["status"] == "ok"
    r = tools.research_question(str(tmp_path), "Foodmandu market share")
    assert r["status"] == "ok" and r["data_gaps"] == ["size"] and r["findings"] == ["f1"]
    assert len(list((tmp_path / "memory").glob("*.json"))) == 2

    c = tools.compile_research(str(tmp_path), remaining_gaps=["market size"])
    assert c["findings"] == 1 and c["sources"] == 3  # deduplicated across rounds
    research = Research.model_validate_json((tmp_path / "research.json").read_text())
    assert research.data_gaps == ["market size"] and (tmp_path / "research.md").is_file()

    w = tools.write_brief(str(tmp_path), iterations=1, evaluate=True)
    assert w["status"] == "ok" and w["verdict"] == "RAG WORKING"
    out = tmp_path / "output"
    assert (out / "final.md").is_file() and (out / "guideline.md").is_file()
    assert json.loads((out / "state.json").read_text())["query"] == "Food delivery in Nepal"


def test_mcp_tools_report_missing_inputs(tmp_path):
    assert tools.compile_research(str(tmp_path))["status"] == "error"
    assert tools.write_brief(str(tmp_path))["status"] == "error"
    assert tools.read_links(str(tmp_path))["status"] == "skipped"


def test_f1_scores_treat_fail_as_positive():
    s = calibration.scores(["fail", "fail", "pass", "pass"], ["fail", "pass", "fail", "pass"])
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 1, 1, 1) and s["f1"] == 0.5
    assert calibration.scores(["pass"], ["pass"])["f1"] == 1.0


def test_split_assignment_is_deterministic():
    ids = [f"run-{i}" for i in range(200)]
    splits = [calibration.assign_split(i) for i in ids]
    assert splits == [calibration.assign_split(i) for i in ids]
    assert {"train", "dev", "test"} == set(splits)


def test_label_and_calibrate(monkeypatch, tmp_path):
    setup(monkeypatch)
    monkeypatch.setattr(config, "EVAL_DATA_DIR", str(tmp_path / "eval"))
    runs = []
    for q in ("good one", "bad one", "train one"):
        st = workflow.orchestrator(q, save_dir=str(tmp_path / "runs"))
        runs.append(st.run_dir)
    calibration.add_label(runs[0], "pass", "Grounded and specific.", split="dev")
    calibration.add_label(runs[1], "fail", "Invents a market size.", split="dev")
    calibration.add_label(runs[2], "fail", "Ignores the angle.", split="train")
    calibration.add_label(runs[2], "fail", "Ignores the angle, relabelled.", split="train")  # replaces
    assert len(calibration.load_dataset()) == 3

    seen = []

    def judge_ask(schema, system, user, model=None, temperature=0.3):
        seen.append(user)
        return calibration.Verdict(critique="looks fine", label="pass")  # a judge that always passes

    monkeypatch.setattr(calibration, "ask_json", judge_ask)
    rep = calibration.calibrate("dev")
    assert rep["n"] == 2 and rep["recall"] == 0.0 and rep["f1"] == 0.0 and rep["accuracy"] == 0.5
    assert "relabelled" in seen[0] and "EXPERT LABEL: fail" in seen[0]  # train row used as few-shot
    assert (tmp_path / "eval" / "calibration-dev.json").is_file()
    with pytest.raises(ValueError):
        calibration.calibrate("train")
    with pytest.raises(ValueError):
        calibration.calibrate("test")  # nothing labelled there yet


def test_mcp_server_registers_tools_prompt_and_resources():
    server = pytest.importorskip("market_research.mcp_server")
    assert "write_brief" in server.WORKFLOW
    assert "research_question" in server.research_workflow("/tmp/x")
    assert json.loads(server.settings())["server"] == "market-research"
    assert "Terminology" in server.profile("terminology")
    server.build_server()  # registration does not raise
