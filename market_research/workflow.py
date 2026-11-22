"""Orchestrator: a LangGraph state machine.

Researcher -> Analyst (v0) -> [Critic -> Analyst edits]* -> Evaluator

The Critic/Analyst pair is an evaluator-optimizer loop with a fixed budget of `iterations` rounds: it
stops early when the Critic approves, otherwise after the last edit. Every version is kept, because
writing quality is subjective and an earlier version is sometimes the better one.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from . import config
from .agents import analyst, critic, researcher
from .llm import usage_snapshot
from .guideline import Guideline
from .models import Research, StepMetric, WorkflowState

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
        if s.research is not None:  # research done elsewhere (for example by the MCP research tools)
            step(s, f"Researcher: using existing research ({len(s.research.findings)} findings).")
            return {"state": s}
        step(s, "Researcher: gathering information" + (" (sample data)..." if g["use_sample"] else " (web search)..."))
        s.research = researcher(s, g["use_sample"])
        for n, qs in enumerate(s.research.gap_rounds, 1):
            step(s, f"Researcher: gap round {n}, searched: " + "; ".join(qs))
        step(s, f"Researcher: {len(s.research.findings)} findings from {len(s.research.sources)} sources, "
                f"{len(s.research.data_gaps)} gaps left.")
        return {"state": s}

    def analyst_node(g: GraphState) -> dict:
        s = g["state"]
        step(s, "Analyst: writing draft brief (v0)...")
        s.draft = analyst(s)
        s.drafts = [s.draft]
        return {"state": s}

    def critic_node(g: GraphState) -> dict:
        s = g["state"]
        v = len(s.drafts) - 1
        step(s, f"Critic: reviewing draft v{v} in a fresh context...")
        c = critic(s, s.drafts[-1])
        if g["force_revision"] and c.approved and v == 0:  # test hook for the revision path
            c.approved = False
            c.revision_instructions = "Tighten wording (forced revision for testing)."
        s.critique = c
        s.critiques.append(c)
        if not c.approved:
            step(s, f"Critic: {len(c.issues)} issues (" + ", ".join(
                sorted({f"{i.rule}/{i.category}" for i in c.issues})) + ")" if c.issues
                else "Critic: revision requested.")
        return {"state": s}

    def approve_node(g: GraphState) -> dict:
        s = g["state"]
        step(s, f"Critic: approved draft v{len(s.drafts) - 1}.")
        s.final = s.drafts[-1]
        return {"state": s}

    def revise_node(g: GraphState) -> dict:
        s = g["state"]
        n = len(s.drafts)
        step(s, f"Analyst: revising, applying {len(s.critique.issues)} issues -> v{n} "
                f"(round {n} of max {s.max_iterations})...")
        new = analyst(s, revise=True, draft=s.drafts[-1], critique=s.critique)
        s.drafts.append(new)
        s.final = new
        s.revised = True
        return {"state": s}

    def stop_node(g: GraphState) -> dict:
        s = g["state"]
        step(s, f"Orchestrator: review budget used ({s.max_iterations} rounds); v{len(s.drafts) - 1} is final.")
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

    def after_revise(g: GraphState) -> str:
        s = g["state"]
        return "critic" if len(s.drafts) - 1 < s.max_iterations else "stop"

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
                     ("approve", approve_node), ("revise", revise_node), ("stop", stop_node),
                     ("evaluate", evaluate_node), ("finish", finish_node)]:
        graph.add_node(name, timed(name, fn))
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "critic")
    graph.add_conditional_edges("critic", after_critic, {"approve": "approve", "revise": "revise"})
    graph.add_conditional_edges("approve", after_final, {"evaluate": "evaluate", "finish": "finish"})
    graph.add_conditional_edges("revise", after_revise, {"critic": "critic", "stop": "stop"})
    graph.add_conditional_edges("stop", after_final, {"evaluate": "evaluate", "finish": "finish"})
    graph.add_edge("evaluate", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def orchestrator(
    query: str = "",
    use_sample: bool = False,
    on_step: Optional[StepCallback] = None,
    evaluate_rag: bool = False,
    force_revision: bool = False,
    ragas: bool = False,
    save_dir: Optional[str] = None,
    guideline: Optional[Guideline] = None,
    iterations: int = config.REVIEW_ITERATIONS,
    research: Optional[Research] = None,
) -> WorkflowState:
    """Run the workflow on a short query, or on a guideline (its topic then replaces the query)."""
    query = (guideline.topic if guideline else query).strip()
    if not query:
        raise ValueError("Please enter a company or industry.")
    if len(query) > config.MAX_QUERY_CHARS:
        raise ValueError(f"Query is too long (max {config.MAX_QUERY_CHARS} characters).")
    iterations = max(1, min(int(iterations), config.MAX_REVIEW_ITERATIONS))

    result = build_graph(on_step).invoke({
        "state": WorkflowState(query=query, guideline=guideline, max_iterations=iterations,
                               research=research), "use_sample": use_sample, "evaluate_rag": evaluate_rag,
        "ragas": ragas and evaluate_rag, "force_revision": force_revision,
    })
    state = result["state"]
    if save_dir:
        from .artifacts import save_run

        save_run(state, save_dir)
        if on_step:
            on_step(f"Orchestrator: run files saved to {state.run_dir}")
    return state
