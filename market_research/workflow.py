"""Orchestrator: a LangGraph state machine Researcher -> Analyst -> Critic (-> Analyst once) -> Evaluator."""
from __future__ import annotations

import time
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from . import config
from .agents import analyst, critic, researcher
from .llm import usage_snapshot
from .models import StepMetric, WorkflowState

StepCallback = Callable[[str], None]


class GraphState(TypedDict):
    state: WorkflowState
    use_sample: bool
    evaluate_rag: bool
    ragas: bool
    force_revision: bool


def build_graph(on_step: Optional[StepCallback] = None):
    """Compile the workflow graph. `on_step` receives each progress message."""

    def step(state: WorkflowState, msg: str) -> None:
        state.log.append(msg)
        if on_step:
            on_step(msg)

    def research_node(g: GraphState) -> dict:
        s = g["state"]
        step(s, f"Orchestrator: received query '{s.query}', starting workflow.")
        step(s, "Researcher: gathering information" + (" (sample data)..." if g["use_sample"] else " (web search)..."))
        s.research = researcher(s, g["use_sample"])
        for n, qs in enumerate(s.research.gap_rounds, 1):
            step(s, f"Researcher: gap round {n}, searched: " + "; ".join(qs))
        step(s, f"Researcher: {len(s.research.findings)} findings from {len(s.research.sources)} sources, "
                f"{len(s.research.data_gaps)} gaps left.")
        return {"state": s}

    def analyst_node(g: GraphState) -> dict:
        s = g["state"]
        step(s, "Analyst: writing draft brief...")
        s.draft = analyst(s)
        return {"state": s}

    def critic_node(g: GraphState) -> dict:
        s = g["state"]
        step(s, "Critic: reviewing draft...")
        s.critique = critic(s)
        if not s.critique.approved:
            step(s, f"Critic: {len(s.critique.issues)} issues " + ", ".join(
                sorted({i.category for i in s.critique.issues})) if s.critique.issues else "Critic: revision requested.")
        if g["force_revision"] and s.critique.approved:  # test hook for the one-revision path
            s.critique.approved = False
            s.critique.revision_instructions = "Tighten wording (forced revision for testing)."
        return {"state": s}

    def approve_node(g: GraphState) -> dict:
        s = g["state"]
        step(s, "Critic: approved, no revision needed.")
        s.final = s.draft
        return {"state": s}

    def revise_node(g: GraphState) -> dict:
        s = g["state"]
        step(s, "Critic: revision requested. Analyst: revising (max one revision)...")
        s.final = analyst(s, revise=True)
        s.revised = True
        return {"state": s}

    def evaluate_node(g: GraphState) -> dict:
        from .evaluation import evaluate

        s = g["state"]
        step(s, "Evaluator: scoring retrieval and grounding...")
        s.evaluation = evaluate(s, use_ragas=g["ragas"])
        step(s, f"Evaluator: {s.evaluation.verdict}.")
        return {"state": s}

    def finish_node(g: GraphState) -> dict:
        step(g["state"], "Orchestrator: workflow complete.")
        return {"state": g["state"]}

    def after_critic(g: GraphState) -> str:
        return "approve" if g["state"].critique.approved else "revise"

    def after_final(g: GraphState) -> str:
        return "evaluate" if g["evaluate_rag"] else "finish"

    def timed(name: str, fn):
        """Record wall time and LLM usage of one node on the shared state."""
        def wrapper(g: GraphState) -> dict:
            t0, u0 = time.perf_counter(), usage_snapshot()
            out = fn(g)
            u1 = usage_snapshot()
            g["state"].metrics.append(StepMetric(
                step=name, seconds=round(time.perf_counter() - t0, 2), llm_calls=u1["calls"] - u0["calls"],
                input_tokens=u1["input_tokens"] - u0["input_tokens"],
                output_tokens=u1["output_tokens"] - u0["output_tokens"]))
            return out
        return wrapper

    graph = StateGraph(GraphState)
    for name, fn in [("researcher", research_node), ("analyst", analyst_node), ("critic", critic_node),
                     ("approve", approve_node), ("revise", revise_node),
                     ("evaluate", evaluate_node), ("finish", finish_node)]:
        graph.add_node(name, timed(name, fn))
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "critic")
    graph.add_conditional_edges("critic", after_critic, {"approve": "approve", "revise": "revise"})
    graph.add_conditional_edges("approve", after_final, {"evaluate": "evaluate", "finish": "finish"})
    graph.add_conditional_edges("revise", after_final, {"evaluate": "evaluate", "finish": "finish"})
    graph.add_edge("evaluate", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def orchestrator(
    query: str,
    use_sample: bool = False,
    on_step: Optional[StepCallback] = None,
    evaluate_rag: bool = False,
    force_revision: bool = False,
    ragas: bool = False,
    save_dir: Optional[str] = None,
) -> WorkflowState:
    query = query.strip()
    if not query:
        raise ValueError("Please enter a company or industry.")
    if len(query) > config.MAX_QUERY_CHARS:
        raise ValueError(f"Query is too long (max {config.MAX_QUERY_CHARS} characters).")

    result = build_graph(on_step).invoke({
        "state": WorkflowState(query=query), "use_sample": use_sample, "evaluate_rag": evaluate_rag,
        "ragas": ragas and evaluate_rag, "force_revision": force_revision,
    })
    state = result["state"]
    if save_dir:
        from .artifacts import save_run

        save_run(state, save_dir)
        if on_step:
            on_step(f"Orchestrator: run files saved to {state.run_dir}")
    return state
