"""Drive the real Streamlit app headlessly (LLM and search mocked) and check the screens a reviewer sees."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.test_workflow_mock import setup

APP = Path(__file__).resolve().parent.parent / "app.py"


def run_app(monkeypatch, sample: bool) -> AppTest:
    setup(monkeypatch)
    monkeypatch.setenv("USE_SAMPLE_DATA", "1" if sample else "0")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Analyze the food delivery market in Nepal").click()
    return at.run()


def test_example_button_runs_all_roles_and_shows_tabs(monkeypatch, tmp_path):
    monkeypatch.setattr("market_research.config.RUNS_DIR", str(tmp_path))
    at = run_app(monkeypatch, sample=False)
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == ["Brief", "Evidence", "Agent trace", "Quality"]
    state = at.session_state["state"]
    assert state.final and state.research.mode == "web"
    assert [t.role for t in state.trace][1:4] == ["Researcher", "Analyst", "Critic"]
    assert not at.warning or all("SAMPLE" not in w.value for w in at.warning)


def test_sample_mode_shows_a_warning_banner(monkeypatch, tmp_path):
    monkeypatch.setattr("market_research.config.RUNS_DIR", str(tmp_path))
    at = run_app(monkeypatch, sample=True)
    assert not at.exception
    assert any("SAMPLE DATA MODE" in w.value for w in at.warning)
    assert at.session_state["state"].research.mode == "sample"
