"""Live agent pipeline for the UI: which role is working, which are done, how long each took.

Pure Python (no Streamlit), so it can be tested. The workflow reports progress as messages such as
"Researcher: gap round 1, searched: ..."; `AgentTracker.update` turns them into per-role status, and
`render_html` draws the pipeline: a pulsing card for the active role, a tick for finished roles, a cross for
the role that failed. The CSS uses only translucent greys and fixed accents, so it reads in light and dark themes.
"""
from __future__ import annotations

import html
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["waiting", "active", "done", "error", "skipped"]

ROLE_INFO = {
    "Orchestrator": ("🧭", "Checks the query and runs the roles"),
    "Researcher": ("🔎", "Searches the web and extracts findings"),
    "Analyst": ("✍️", "Writes the brief from the findings"),
    "Critic": ("🧐", "Reviews the draft, asks for one revision"),
    "Evaluator": ("📊", "Scores retrieval and grounding"),
}


@dataclass
class Stage:
    role: str
    status: Status = "waiting"
    detail: str = ""
    seconds: float = 0.0
    runs: int = 0
    _started: float | None = None

    def start(self, now: float) -> None:
        if self.status != "active":
            self.status, self._started, self.runs = "active", now, self.runs + 1

    def stop(self, now: float, status: Status = "done") -> None:
        if self.status == "active" and self._started is not None:
            self.seconds += now - self._started
        self.status, self._started = status, None

    def elapsed(self, now: float) -> float:
        return self.seconds + (now - self._started if self._started is not None else 0.0)


@dataclass
class AgentTracker:
    roles: list[str]
    clock: Callable[[], float] = time.monotonic
    stages: dict[str, Stage] = field(init=False)
    feed: list[tuple[float, str, str]] = field(default_factory=list)  # (seconds since start, role, text)
    started: float = field(init=False)
    finished: bool = False
    failed: bool = False

    def __post_init__(self) -> None:
        self.stages = {r: Stage(r) for r in self.roles}
        self.started = self.clock()

    @property
    def active(self) -> str | None:
        return next((r for r, s in self.stages.items() if s.status == "active"), None)

    def update(self, message: str) -> None:
        """Feed one progress message ('Role: text'). Unknown prefixes only go to the activity feed."""
        now = self.clock()
        role, _, text = message.partition(":")
        role, text = role.strip(), text.strip()
        if role not in self.stages or self.finished:  # after the run, messages only go to the log
            self.feed.append((now - self.started, role if role in self.stages else "", text or message.strip()))
            return
        current = self.active
        if current and current != role:
            self.stages[current].stop(now)
        stage = self.stages[role]
        stage.start(now)
        stage.detail = text
        self.feed.append((now - self.started, role, text))
        if role == "Orchestrator" and text.startswith("workflow complete"):
            self.finish()

    def finish(self) -> None:
        now = self.clock()
        for s in self.stages.values():
            if s.status == "active":
                s.stop(now)
            elif s.status == "waiting":
                s.status = "skipped"
        self.finished = True

    def fail(self, message: str = "") -> None:
        """Mark the active role (or the first unfinished one) as failed."""
        now = self.clock()
        role = self.active or next((r for r, s in self.stages.items() if s.status == "waiting"), self.roles[-1])
        self.stages[role].stop(now, "error")
        if message:
            self.stages[role].detail = message
        self.finished = self.failed = True

    def total(self) -> float:
        return self.clock() - self.started

    def done_count(self) -> int:
        return sum(s.status in ("done", "skipped") for s in self.stages.values())


def _fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _short(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


CSS = """
<style>
.mr-pipe{display:grid;gap:.6rem;margin:.25rem 0 .75rem}
.mr-card{min-width:0;border:1px solid rgba(128,128,128,.30);border-radius:.75rem;
 padding:.55rem .6rem;background:rgba(128,128,128,.06);position:relative;transition:all .3s ease}
.mr-card:not(:last-child)::after{content:"➜";position:absolute;right:-.58rem;top:50%;transform:translateY(-50%);
 font-size:.62rem;opacity:.35;z-index:1}
.mr-card.mr-lit:not(:last-child)::after{opacity:.9;color:#16a34a}
.mr-head{display:flex;align-items:center;gap:.35rem;font-weight:600;font-size:.86rem;min-width:0}
.mr-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
@media (max-width:640px){.mr-pipe{grid-template-columns:repeat(2,minmax(0,1fr))!important}
 .mr-card::after{display:none}}
.mr-ico{font-size:1.05rem}
.mr-badge{margin-left:auto;width:1.35rem;height:1.35rem;border-radius:50%;display:flex;align-items:center;
 justify-content:center;font-size:.8rem;font-weight:700;flex:none}
.mr-sub{font-size:.74rem;opacity:.7;margin-top:.3rem;line-height:1.25;min-height:1.9em}
.mr-time{font-size:.72rem;opacity:.75;margin-top:.3rem;font-variant-numeric:tabular-nums}
.mr-waiting{opacity:.55}
.mr-waiting .mr-badge{border:2px solid rgba(128,128,128,.45)}
.mr-skipped{opacity:.4}
.mr-skipped .mr-badge{border:2px dashed rgba(128,128,128,.45)}
.mr-active{border-color:#3b82f6;background:rgba(59,130,246,.10);animation:mr-glow 1.4s ease-in-out infinite}
.mr-active .mr-badge{border:2.5px solid rgba(59,130,246,.25);border-top-color:#3b82f6;animation:mr-spin .8s linear infinite}
.mr-active .mr-ico{animation:mr-blink 1.1s ease-in-out infinite}
.mr-done{border-color:rgba(22,163,74,.55);background:rgba(22,163,74,.08)}
.mr-done .mr-badge{background:#16a34a;color:#fff;animation:mr-pop .35s ease-out}
.mr-error{border-color:#dc2626;background:rgba(220,38,38,.10)}
.mr-error .mr-badge{background:#dc2626;color:#fff}
.mr-bar{height:6px;border-radius:3px;background:rgba(128,128,128,.18);overflow:hidden;margin:.35rem 0 .2rem}
.mr-fill{height:100%;background:linear-gradient(90deg,#3b82f6,#16a34a);transition:width .5s ease}
.mr-run .mr-fill{background-size:200% 100%;background-image:linear-gradient(90deg,#3b82f6,#60a5fa,#3b82f6);
 animation:mr-slide 1.5s linear infinite}
.mr-top{display:flex;justify-content:space-between;font-size:.85rem;font-weight:600;align-items:center}
.mr-top .mr-state{font-weight:500;opacity:.8}
.mr-feed{font-size:.78rem;border-left:2px solid rgba(128,128,128,.3);padding-left:.6rem;margin-top:.4rem;
 max-height:11.5rem;overflow:auto}
.mr-row{display:flex;gap:.5rem;padding:.12rem 0;line-height:1.3}
.mr-row:last-child{font-weight:600}
.mr-t{opacity:.55;font-variant-numeric:tabular-nums;flex:none;width:3.2rem}
.mr-r{flex:none;width:5.6rem;opacity:.85}
@keyframes mr-spin{to{transform:rotate(360deg)}}
@keyframes mr-blink{0%,100%{opacity:1}50%{opacity:.25}}
@keyframes mr-glow{0%,100%{box-shadow:0 0 0 0 rgba(59,130,246,.0)}50%{box-shadow:0 0 0 4px rgba(59,130,246,.22)}}
@keyframes mr-pop{0%{transform:scale(.3)}70%{transform:scale(1.2)}100%{transform:scale(1)}}
@keyframes mr-slide{to{background-position:-200% 0}}
@media (prefers-reduced-motion:reduce){.mr-active,.mr-active .mr-badge,.mr-active .mr-ico,.mr-done .mr-badge,
 .mr-run .mr-fill{animation:none}}
</style>
"""

_BADGE = {"waiting": "", "active": "", "done": "✓", "error": "✕", "skipped": ""}


def render_html(t: AgentTracker, feed_lines: int = 8) -> str:
    """Progress bar, one card per role and the latest activity. Single-line HTML (safe for st.markdown)."""
    now = t.clock()
    n = len(t.stages)
    if t.failed:
        label, cls = "Stopped early", ""
    elif t.finished:
        label, cls = f"Done in {_fmt(t.total())}", ""
    else:
        label, cls = (f"{t.active} is working…" if t.active else "Starting…"), " mr-run"
    share = 100 if t.finished and not t.failed else int(100 * (t.done_count() + 0.5 * bool(t.active)) / n)
    clock_text = "" if t.finished and not t.failed else f" · {_fmt(t.total())}"  # "Done in …" already has it
    parts = [CSS, f'<div class="mr-top"><span>Agent pipeline</span><span class="mr-state">{html.escape(label)}'
             f'{clock_text}</span></div>',
             f'<div class="mr-bar{cls}"><div class="mr-fill" style="width:{share}%"></div></div><div class="mr-pipe" style="grid-template-columns:repeat({n},minmax(0,1fr))">']
    for role, s in t.stages.items():
        icon, blurb = ROLE_INFO.get(role, ("•", ""))
        sub = _short(s.detail, 70) if s.detail and s.status in ("active", "error") else blurb
        if s.status == "waiting":
            when = "waiting"
        elif s.status == "skipped":
            when = "not run"
        else:
            when = _fmt(s.elapsed(now)) + (f" · ×{s.runs}" if s.runs > 1 else "")
        parts.append(
            f'<div class="mr-card mr-{s.status}{" mr-lit" if s.status == "done" else ""}" '
            f'title="{html.escape(s.detail or blurb)}">'
            f'<div class="mr-head"><span class="mr-ico">{icon}</span><span class="mr-name">{html.escape(role)}</span>'
            f'<span class="mr-badge">{_BADGE[s.status]}</span></div>'
            f'<div class="mr-sub">{html.escape(sub)}</div><div class="mr-time">{when}</div></div>')
    parts.append("</div>")
    if feed_lines:
        parts.append(_feed(t, feed_lines))
    return "".join(parts).replace("\n", "")


def _feed(t: AgentTracker, lines: int) -> str:
    if not t.feed:
        return ""
    rows = "".join(
        f'<div class="mr-row"><span class="mr-t">{_fmt(sec)}</span>'
        f'<span class="mr-r">{html.escape(role)}</span><span>{html.escape(_short(text, 140))}</span></div>'
        for sec, role, text in t.feed[-lines:])
    return f'<div class="mr-feed">{rows}</div>'


def render_feed_html(t: AgentTracker) -> str:
    """The whole activity log, with its styles, for the 'Activity log' expander."""
    return (CSS + _feed(t, len(t.feed))).replace("\n", "")
