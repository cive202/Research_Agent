"""Streamlit UI for the multi-agent market research app (entry point: `streamlit run app.py`)."""
from pathlib import Path

import streamlit as st

from market_research import config, orchestrator
from market_research.guideline import parse_guideline
from market_research.llm import LLMError, explain_error
from market_research.models import EvaluationReport, WorkflowState
from market_research.progress import AgentTracker, render_feed_html, render_html
from market_research.rendering import (
    SAMPLE_WARNING,
    brief_to_markdown,
    link_citations,
    source_citations,
    state_to_json,
)
from market_research.urls import normalize_url

EXAMPLES = [
    "Analyze the food delivery market in Nepal",
    "Analyze the ride-hailing market in Nepal",
    "Analyze Pathao as a company",
]
ROLES = ["Orchestrator", "Researcher", "Analyst", "Critic"]
CLARIFY_PREFIX = "One question before I start:"


def render_sidebar() -> tuple[bool, bool, bool, int]:
    with st.sidebar:
        st.header("Options")
        use_sample = st.checkbox(
            "Use sample data instead of web search",
            value=config.use_sample_env(),
            help="Skips web search and uses fixed, clearly labelled sample data (about Nepal food delivery).",
        )
        run_eval = st.checkbox(
            "Run RAG evaluation",
            value=True,
            help="Scores retrieval relevance, groundedness and citation validity (extra LLM calls).",
        )
        run_ragas = st.checkbox(
            "Also compute RAGAS metrics",
            value=False,
            disabled=not run_eval,
            help="Adds RAGAS faithfulness and context precision (informational, many extra LLM calls).",
        )
        gap_rounds = st.slider(
            "Gap-filling search rounds",
            min_value=0, max_value=3, value=config.MAX_GAP_ROUNDS,
            disabled=use_sample,
            help="After the first pass, the Researcher searches again for facts it could not find (for example "
                 "market size). More rounds give fuller briefs but take longer and use more Groq tokens. "
                 "0 turns it off.",
        )
        provider = "Tavily" if config.tavily_api_key() else "DuckDuckGo (no key needed)"
        st.caption(f"Search provider: {'sample data' if use_sample else provider}. "
                   f"The Critic may request at most {config.MAX_REVIEW_ITERATIONS} revision.")
        if not config.groq_api_key():
            st.warning("GROQ_API_KEY is not set. Copy `.env.example` to `.env` and add your key (see README).")
    return use_sample, run_eval, run_ragas and run_eval, gap_rounds


def draw(slot, tracker: AgentTracker, feed_lines: int = 8) -> None:
    """Redraw the pipeline: pulsing card for the working role, ticks for finished ones, live activity feed."""
    slot.markdown(render_html(tracker, feed_lines), unsafe_allow_html=True)


def run_workflow(query: str, use_sample: bool, run_eval: bool, ragas: bool = False,
                 guideline_text: str = "", gap_rounds: int = config.MAX_GAP_ROUNDS) -> None:
    guideline = None
    if guideline_text.strip():
        try:
            guideline = parse_guideline(guideline_text)
        except ValueError as e:
            st.warning(str(e))
            return
    if not query.strip() and not guideline:
        st.warning("Enter a company or industry first, or click an example.")
        return
    st.session_state.pop("state", None)  # never show an older run next to a new failure
    st.session_state.pop("tracker", None)
    tracker = AgentTracker(ROLES + (["Evaluator"] if run_eval else []))
    slot = st.empty()
    tracker.update("Orchestrator: checking the query...")
    draw(slot, tracker)

    def on_step(msg: str) -> None:
        tracker.update(msg)
        draw(slot, tracker)

    try:
        state = orchestrator(query, use_sample=use_sample, on_step=on_step, evaluate_rag=run_eval,
                             ragas=ragas, save_dir=config.RUNS_DIR, guideline=guideline,
                             raise_errors=False, max_gap_rounds=gap_rounds)
        if state.error:
            tracker.fail(state.error)
        elif not tracker.finished:
            tracker.finish()
        st.session_state["state"] = state
    except (LLMError, ValueError) as e:  # bad input or a problem the user can fix
        tracker.fail("needs clarification" if str(e).startswith(CLARIFY_PREFIX) else str(e))
        (st.info if str(e).startswith(CLARIFY_PREFIX) else st.error)(str(e))
    except Exception as e:  # never show a raw stack trace
        tracker.fail(explain_error(e))
        st.error(explain_error(e))
    draw(slot, tracker, feed_lines=0)  # the finished pipeline; the full log moves into an expander
    st.session_state["tracker"] = tracker


def render_brief(state: WorkflowState) -> None:
    brief = state.final
    st.subheader(f"Brief: {state.query}")
    st.markdown("**Summary**")
    r = state.research
    st.markdown(link_citations(brief.summary, r))

    if brief.competitors:
        st.markdown("**Competitors** (at most two)")
        for c in brief.competitors:
            st.markdown(f"- **{c.name}**: {link_citations(c.note, r)}")

    st.markdown("**Key observations**")
    for o in brief.key_observations:
        st.markdown(f"- {link_citations(o, r)}")

    st.markdown("**Sources**")
    if brief.sources:
        cites = source_citations(brief, r)
        for s in brief.sources:
            ids = cites.get(normalize_url(s.url), [])
            tag = f" (cited as {', '.join(ids)})" if ids else ""
            st.markdown(f"- [{s.title}]({s.url}){tag}" if s.url else f"- {s.title}")
    else:
        st.markdown("- None")

    st.markdown("**Missing information**")
    for m in brief.missing_information or ["None noted"]:
        st.markdown(f"- {m}")


def render_evidence(state: WorkflowState) -> None:
    r = state.research
    st.caption(f"{len(r.findings)} numbered findings from {len(r.sources)} retrieved documents "
               f"(provider: {r.provider or r.mode}). The brief may only cite these IDs.")
    if r.focus:
        background = sum(not f.on_focus for f in r.findings)
        st.caption(f"Focus: **{', '.join(r.focus)}**. {background} finding(s) do not mention it and are "
                   "marked background: the brief may use them only as general context.")
    rows = [{"ID": f.id, "Focus": "yes" if f.on_focus else "background", "Claim": f.claim,
             "Source": f.source_url or f.source_title or "n/a", "Snippet": f.snippet} for f in r.findings]
    st.dataframe(rows, hide_index=True, width="stretch", column_config={
        "Source": st.column_config.LinkColumn("Source", display_text=r"https?://(?:www\.)?([^/]+)"),
    })
    if r.data_gaps:
        st.markdown("**Data gaps the Researcher reported**")
        for g in r.data_gaps:
            st.markdown(f"- {g}")


def render_trace(state: WorkflowState) -> None:
    st.markdown("**Run trace**")
    st.table([{"Role": t.role, "Status": t.status, "Seconds": t.seconds, "Detail": t.detail} for t in state.trace])
    revised = "yes, one revision" if state.revised else "no revision needed"
    verdict = "approved" if state.critique and state.critique.approved else "changes requested"
    st.markdown(f"**Critic verdict:** {verdict}. **Revision happened:** {revised}. "
                f"{len(state.drafts)} version(s) of the brief.")
    st.markdown("**Grounding flags on the final brief**")
    if state.grounding:
        for f in state.grounding:
            st.warning(f"`{f.code}` ({f.section}): {f.message}")
    else:
        st.success("No flags: every observation and competitor cites an existing finding.")

    st.markdown("**Handoffs (structured data passed between roles)**")
    with st.expander("Orchestrator → Researcher: query"):
        st.json({"query": state.query, "guideline": bool(state.guideline), "max_revisions": state.max_iterations})
    with st.expander("Researcher → Analyst: findings"):
        st.json(state.research.model_dump(exclude={"sources"}), expanded=False)
    with st.expander("Analyst → Critic: draft"):
        st.json(state.drafts[0].model_dump(), expanded=False)
    with st.expander("Critic → Analyst: feedback"):
        for n, c in enumerate(state.critiques):
            st.markdown(f"Review of v{n}: " + ("approved" if c.approved else f"{len(c.issues)} issue(s)"))
            for i in c.issues:
                st.markdown(f"- `{i.rule}/{i.category}` in *{i.section}*: {i.comment}")
        if not state.critiques:
            st.markdown("No review recorded.")
    if len(state.drafts) > 1:
        with st.expander("Versions of the brief"):
            tabs = st.tabs([f"v{n}" for n in range(len(state.drafts))])
            for tab, d in zip(tabs, state.drafts):
                with tab:
                    st.markdown(brief_to_markdown(state, d))
    with st.expander("Full shared state (JSON)"):
        st.json(state.model_dump(mode="json"), expanded=False)


def render_evaluation(ev: EvaluationReport) -> None:
    if ev.verdict == "RAG WORKING":
        st.success(f"Verdict: {ev.verdict}")
    elif ev.verdict == "RAG NOT ACTIVE":
        st.info(f"Verdict: {ev.verdict} (sample data, so retrieval was not exercised)")
    else:
        st.error(f"Verdict: {ev.verdict}")
    st.table([{"Metric": m.name, "Score": f"{m.score:.2f}", "Pass": "yes" if m.passed else "no",
               "Reason": m.reason} for m in ev.metrics])
    if ev.failed_checks:
        st.markdown("**Failed checks:** " + ", ".join(ev.failed_checks))
    for m in ev.metrics:
        for d in m.details:
            st.markdown(f"- *{m.name}*: {d}")


def render_quality(state: WorkflowState) -> None:
    st.markdown("**Is the evidence good enough?**")
    st.markdown(f"- Findings: {len(state.research.findings)} (thin below {config.THIN_EVIDENCE_FINDINGS})")
    st.markdown(f"- Grounding flags: {len(state.grounding)}")
    if state.evaluation:
        render_evaluation(state.evaluation)
    else:
        st.caption("RAG evaluation was switched off in the sidebar.")
    if state.metrics:
        st.markdown("**Run cost**")
        st.table([{"Step": m.step, "Seconds": m.seconds, "LLM calls": m.llm_calls,
                   "Tokens in": m.input_tokens, "Tokens out": m.output_tokens} for m in state.metrics])
    if state.run_dir:
        st.caption(f"Run files saved to `{state.run_dir}`")


def render_downloads(state: WorkflowState) -> None:
    c1, c2 = st.columns(2)
    c1.download_button("Download brief (Markdown)", brief_to_markdown(state),
                       file_name="research_brief.md", mime="text/markdown")
    c2.download_button("Download run (JSON)", state_to_json(state),
                       file_name="research_run.json", mime="application/json")


def render_activity(tracker: AgentTracker | None, show_pipeline: bool) -> None:
    """The finished pipeline (on reruns, e.g. after switching tabs) and the full activity log."""
    if tracker is None:
        return
    if show_pipeline:
        st.markdown(render_html(tracker, feed_lines=0), unsafe_allow_html=True)
    with st.expander(f"Activity log ({len(tracker.feed)} steps)"):
        st.markdown(render_feed_html(tracker), unsafe_allow_html=True)


def render_metrics(state: WorkflowState) -> None:
    """Headline numbers above the tabs."""
    r = state.research
    seconds = sum(m.seconds for m in state.metrics)
    tokens = sum(m.input_tokens + m.output_tokens for m in state.metrics)
    verdict = state.evaluation.verdict.replace("RAG ", "") if state.evaluation else "not run"
    critic = "approved" if state.critique and state.critique.approved else (
        "revised" if state.revised else "changes asked")
    cols = st.columns(5)
    cols[0].metric("Findings", len(r.findings), help="Numbered facts the brief may cite")
    cols[1].metric("Sources", len(r.sources), help=f"Provider: {r.provider or r.mode}")
    cols[2].metric("Critic", critic, help=f"{len(state.drafts)} version(s) of the brief")
    cols[3].metric("RAG check", verdict.title(), help="Evaluator verdict (Quality tab)")
    cols[4].metric("Run time", f"{int(seconds // 60)}m {int(seconds % 60):02d}s" if seconds >= 60
                   else f"{seconds:.0f}s", help=f"{tokens:,} LLM tokens")


def render_result(state: WorkflowState) -> None:
    render_metrics(state)
    if state.research.mode == "sample":
        st.warning(SAMPLE_WARNING)
    t_brief, t_evidence, t_trace, t_quality = st.tabs(["Brief", "Evidence", "Agent trace", "Quality"])
    with t_brief:
        render_brief(state)
        render_downloads(state)
    with t_evidence:
        render_evidence(state)
    with t_trace:
        render_trace(state)
    with t_quality:
        render_quality(state)


def main() -> None:
    st.set_page_config(page_title="Market Research Agents", page_icon="🔎", layout="centered")
    st.title("Market research brief")
    st.caption("Five roles work in turn: Orchestrator → Researcher → Analyst → Critic (at most one revision) → "
               "Evaluator. Watch them work below once you start.")

    use_sample, run_eval, run_ragas, gap_rounds = render_sidebar()
    if "query" not in st.session_state:
        st.session_state["query"] = ""
    st.markdown("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, text in zip(cols, EXAMPLES):
        if col.button(text, width="stretch"):
            st.session_state["query"] = text
            st.session_state["autorun"] = True
    query = st.text_input("Company or industry", key="query", placeholder=EXAMPLES[0])
    with st.expander("Guideline (optional)"):
        st.caption("Angle, audience, key questions, seed links and constraints. The title line is the topic "
                   "and replaces the query above.")
        template = Path(config.PROJECT_ROOT, "guidelines", "nepal-food-delivery.md")
        guideline_text = st.text_area("Guideline markdown", height=260,
                                      placeholder=template.read_text(encoding="utf-8") if template.is_file() else "")
    started = st.button("Start research", type="primary")
    ran_now = bool(started or st.session_state.pop("autorun", False))
    if ran_now:
        run_workflow(query, use_sample, run_eval, run_ragas, guideline_text, gap_rounds)
    render_activity(st.session_state.get("tracker"), show_pipeline=not ran_now)

    state = st.session_state.get("state")
    if state and state.error:
        st.error(state.error)
        if state.trace:
            st.table([{"Role": t.role, "Status": t.status, "Seconds": t.seconds, "Detail": t.detail}
                      for t in state.trace])
    elif state and state.final:
        render_result(state)


main()
