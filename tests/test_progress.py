"""The live agent pipeline shown in the UI (pure Python, no Streamlit)."""
from market_research.progress import AgentTracker, render_feed_html, render_html

ROLES = ["Orchestrator", "Researcher", "Analyst", "Critic", "Evaluator"]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def tracker():
    clock = Clock()
    return AgentTracker(ROLES, clock=clock), clock


def statuses(t):
    return {r: s.status for r, s in t.stages.items()}


def test_roles_go_active_then_done_with_timings():
    t, clock = tracker()
    t.update("Orchestrator: received query 'x'")
    clock.t = 1
    t.update("Researcher: gathering information (web search)...")
    assert t.active == "Researcher" and statuses(t)["Orchestrator"] == "done"
    clock.t = 31
    t.update("Researcher: gap round 1, searched: a; b")
    t.update("Analyst: writing draft brief (v0)...")
    assert t.stages["Researcher"].seconds == 30 and t.stages["Researcher"].runs == 1
    assert "gap round" not in t.stages["Analyst"].detail
    html = render_html(t)
    assert 'mr-card mr-active' in html and html.count('mr-card mr-done') == 2
    assert "Analyst is working" in html and 'class="mr-bar mr-run"' in html  # animated bar while running


def test_revision_reactivates_analyst_and_completion_ticks_everything():
    t, clock = tracker()
    for msg in ["Orchestrator: received", "Researcher: go", "Analyst: v0", "Critic: reviewing",
                "Analyst: revising -> v1", "Orchestrator: review budget used", "Evaluator: scoring",
                "Orchestrator: workflow complete."]:
        clock.t += 2
        t.update(msg)
    assert t.finished and not t.failed
    t.update("Orchestrator: run files saved to runs/x")  # arrives after completion: log only
    assert set(statuses(t).values()) == {"done"} and t.feed[-1][2].startswith("run files saved")
    assert t.stages["Analyst"].runs == 2
    html = render_html(t)
    header = html[html.index('class="mr-top"'):html.index('class="mr-bar')]
    assert "Done in 16s" in header and header.count("16s") == 1  # the total is shown once
    assert "Done in" in html and "×2" in html and html.count("✓") == 5 and 'class="mr-bar mr-run"' not in html


def test_evaluator_not_reached_is_shown_as_skipped():
    t, _ = tracker()
    for msg in ["Orchestrator: received", "Researcher: go", "Analyst: v0", "Critic: ok", "Orchestrator: workflow complete."]:
        t.update(msg)
    assert statuses(t)["Evaluator"] == "skipped"


def test_failure_marks_the_working_role_with_a_cross():
    t, _ = tracker()
    t.update("Orchestrator: received")
    t.update("Researcher: gathering information")
    t.fail("Groq rate limit hit")
    assert statuses(t)["Researcher"] == "error" and t.failed
    html = render_html(t)
    assert "Stopped early" in html and "✕" in html and "Groq rate limit hit" in html


def test_unknown_messages_and_html_escaping_are_safe():
    t, _ = tracker()
    t.update("no role here <script>")
    t.update("Researcher: <b>bold</b>")
    html = render_html(t)
    assert "<script>" not in html and "&lt;b&gt;bold" in html and "\n" not in html
    assert t.feed[0][1] == "" and "&lt;script&gt;" in render_feed_html(t)
