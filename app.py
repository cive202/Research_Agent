"""Streamlit UI for the multi-agent market research app (entry point: `streamlit run app.py`)."""
from pathlib import Path

import streamlit as st

from market_research import config, orchestrator
from market_research.guideline import parse_guideline
from market_research.llm import LLMError, explain_error
from market_research.models import EvaluationReport, WorkflowState
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


def render_sidebar() -> tuple[bool, bool, bool]:
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
        provider = "Tavily" if config.tavily_api_key() else "DuckDuckGo (no key needed)"
        st.caption(f"Search provider: {'sample data' if use_sample else provider}. "
                   f"The Critic may request at most {config.MAX_REVIEW_ITERATIONS} revision.")
        if not config.groq_api_key():
            st.warning("GROQ_API_KEY is not set. Copy `.env.example` to `.env` and add your key (see README).")
    return use_sample, run_eval, run_ragas and run_eval


def role_progress(slot, message: str, done: set, current: list) -> None:
    """Update the one-line role tracker: finished roles get a tick, the active role a spinner."""
    role = message.split(":", 1)[0]
    if role in ROLES:
        if current and current[0] != role:
            done.add(current[0])
        current[:] = [role]
    icons = {r: "✅" if r in done else ("⏳" if current and current[0] == r else "⬜") for r in ROLES}
    slot.markdown("  ➜  ".join(f"{icons[r]} {r}" for r in ROLES))


def run_workflow(query: str, use_sample: bool, run_eval: bool, ragas: bool = False,
                 guideline_text: str = "") -> None:
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
    done: set = set()
    current: list = []
    try:
        with st.status("Running agents...", expanded=True) as status:
            slot = st.empty()
            role_progress(slot, "", done, current)

            def on_step(msg: str) -> None:
                role_progress(slot, msg, done, current)
                st.write(msg)

            state = orchestrator(query, use_sample=use_sample, on_step=on_step, evaluate_rag=run_eval,
                                 ragas=ragas, save_dir=config.RUNS_DIR, guideline=guideline,
                                 raise_errors=False)
            if state.error:
                status.update(label="Stopped early", state="error")
            else:
                done.update(ROLES)
                role_progress(slot, "", done, [])
                status.update(label="Done", state="complete")
        st.session_state["state"] = state
    except (LLMError, ValueError) as e:  # bad input or a problem the user can fix
        st.error(str(e))
    except Exception as e:  # never show a raw stack trace
        st.error(explain_error(e))


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
    rows = [{"ID": f.id, "Claim": f.claim, "Source": f.source_url or f.source_title or "n/a",
             "Snippet": f.snippet} for f in r.findings]
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


def render_result(state: WorkflowState) -> None:
    st.divider()
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
    st.caption("Orchestrator → Researcher → Analyst → Critic (at most one revision) → Evaluator.")

    use_sample, run_eval, run_ragas = render_sidebar()
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
    if started or st.session_state.pop("autorun", False):
        run_workflow(query, use_sample, run_eval, run_ragas, guideline_text)

    state = st.session_state.get("state")
    if state and state.error:
        st.error(state.error)
        if state.trace:
            st.table([{"Role": t.role, "Status": t.status, "Seconds": t.seconds, "Detail": t.detail}
                      for t in state.trace])
    elif state and state.final:
        render_result(state)


main()
