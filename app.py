"""Streamlit UI for the multi-agent market research app (entry point: `streamlit run app.py`)."""
import streamlit as st

from pathlib import Path

from market_research import config, orchestrator
from market_research.guideline import parse_guideline
from market_research.llm import LLMError
from market_research.models import EvaluationReport, WorkflowState
from market_research.rendering import brief_to_markdown


def render_sidebar() -> tuple[bool, bool, bool, int]:
    with st.sidebar:
        st.header("Options")
        use_sample = st.checkbox(
            "Use sample data instead of web search",
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
        iterations = st.slider(
            "Review rounds", 1, config.MAX_REVIEW_ITERATIONS, config.REVIEW_ITERATIONS,
            help="Maximum Critic review -> Analyst edit rounds. Stops early when the Critic approves.",
        )
        st.markdown("Set `GROQ_API_KEY` (and optionally `TAVILY_API_KEY`) in your `.env` file. See the README.")
    return use_sample, run_eval, run_ragas and run_eval, iterations


def run_workflow(query: str, use_sample: bool, run_eval: bool, ragas: bool = False,
                 guideline_text: str = "", iterations: int = config.REVIEW_ITERATIONS) -> None:
    guideline = None
    if guideline_text.strip():
        try:
            guideline = parse_guideline(guideline_text)
        except ValueError as e:
            st.warning(str(e))
            return
    if not query.strip() and not guideline:
        st.warning("Enter a company or industry first (or fill in a guideline).")
        return
    try:
        with st.status("Running agents...", expanded=True) as status:
            state = orchestrator(query, use_sample=use_sample, on_step=st.write, evaluate_rag=run_eval, ragas=ragas,
                                 save_dir=config.RUNS_DIR, guideline=guideline, iterations=iterations)
            status.update(label="Done", state="complete")
        st.session_state["state"] = state
    except LLMError as e:
        st.error(str(e))
    except ValueError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")


def render_brief(state: WorkflowState) -> None:
    brief = state.final
    if state.research.mode == "sample":
        st.warning("Sample data mode: findings are fixed illustrative data, not live research.")

    st.subheader(f"Brief: {state.query}")
    st.markdown("**Summary**")
    st.write(brief.summary)

    if brief.competitors:
        st.markdown("**Competitors**")
        for c in brief.competitors:
            st.markdown(f"- **{c.name}**: {c.note}")

    st.markdown("**Key observations**")
    for o in brief.key_observations:
        st.markdown(f"- {o}")

    st.markdown("**Sources**")
    if brief.sources:
        for s in brief.sources:
            st.markdown(f"- [{s.title}]({s.url})" if s.url else f"- {s.title}")
    else:
        st.markdown("- None")

    st.markdown("**Missing information**")
    for m in brief.missing_information or ["None noted"]:
        st.markdown(f"- {m}")


def render_evaluation(ev: EvaluationReport) -> None:
    st.divider()
    st.subheader("Is the RAG working?")
    if ev.verdict == "RAG WORKING":
        st.success(f"Verdict: {ev.verdict}")
    elif ev.verdict == "RAG NOT ACTIVE":
        st.info(f"Verdict: {ev.verdict} (sample data, so retrieval was not exercised)")
    else:
        st.error(f"Verdict: {ev.verdict}")
    st.table(
        [
            {"Metric": m.name, "Score": f"{m.score:.2f}", "Pass": "yes" if m.passed else "no", "Reason": m.reason}
            for m in ev.metrics
        ]
    )
    if ev.failed_checks:
        st.markdown("**Failed checks:** " + ", ".join(ev.failed_checks))
    for m in ev.metrics:
        for d in m.details:
            st.markdown(f"- *{m.name}*: {d}")


def render_handoffs(state: WorkflowState) -> None:
    with st.expander("Agent handoffs (shared data)"):
        verdict = "approved" if state.critique.approved else "changes requested"
        st.markdown(f"**Critic verdict (last round):** {verdict}. "
                    f"{len(state.critiques)} review round(s), {len(state.drafts)} version(s).")
        for n, c in enumerate(state.critiques):
            st.markdown(f"Review of v{n}: " + ("approved" if c.approved else f"{len(c.issues)} issue(s)"))
            for i in c.issues:
                st.markdown(f"- `{i.rule}/{i.category}` in *{i.section}*: {i.comment}")
        if len(state.drafts) > 1:
            tabs = st.tabs([f"v{n}" for n in range(len(state.drafts))])
            for tab, d in zip(tabs, state.drafts):
                with tab:
                    st.markdown(brief_to_markdown(state, d))
        st.json(state.model_dump(), expanded=False)


def render_run_details(state: WorkflowState) -> None:
    with st.expander("Research rounds and run cost"):
        r = state.research
        if r.queries:
            st.markdown("**Initial queries:** " + "; ".join(r.queries))
        for n, qs in enumerate(r.gap_rounds, 1):
            st.markdown(f"**Gap round {n}:** " + "; ".join(qs))
        if not r.gap_rounds and r.mode == "web":
            st.markdown("No follow-up search was needed or possible.")
        st.table([
            {"Step": m.step, "Seconds": m.seconds, "LLM calls": m.llm_calls,
             "Tokens in": m.input_tokens, "Tokens out": m.output_tokens}
            for m in state.metrics
        ])
        if state.run_dir:
            st.caption(f"Run files saved to `{state.run_dir}`")


def render_result(state: WorkflowState) -> None:
    st.divider()
    render_brief(state)
    if state.evaluation:
        render_evaluation(state.evaluation)
    st.download_button(
        "Download brief (Markdown)",
        brief_to_markdown(state),
        file_name="research_brief.md",
        mime="text/markdown",
    )
    render_handoffs(state)
    render_run_details(state)


def main() -> None:
    st.set_page_config(page_title="Market Research Agents", page_icon="🔎", layout="centered")
    st.title("Market research brief")
    st.caption("Orchestrator → Researcher → Analyst → (Critic ⇄ Analyst, up to N rounds) → Evaluator.")

    use_sample, run_eval, run_ragas, iterations = render_sidebar()
    query = st.text_input("Company or industry", placeholder="Analyze the food delivery market in Nepal")
    with st.expander("Guideline (optional, recommended)"):
        st.caption("Angle, audience, key questions, seed links and constraints. The title line is the topic "
                   "and replaces the query above.")
        template = Path(config.PROJECT_ROOT, "guidelines", "nepal-food-delivery.md")
        guideline_text = st.text_area("Guideline markdown", height=260,
                                      placeholder=template.read_text(encoding="utf-8") if template.is_file() else "")
    if st.button("Start research", type="primary"):
        run_workflow(query, use_sample, run_eval, run_ragas, guideline_text, iterations)

    state = st.session_state.get("state")
    if state and state.final:
        render_result(state)


main()
