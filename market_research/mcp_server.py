"""MCP server: exposes the research and writing tools, a workflow prompt and read-only resources.

Run it with `python -m market_research.mcp_server` (stdio). `.mcp.json` in the project root registers it
with Claude Code; any other MCP client (Cursor, Copilot, ...) can use the same command.

  tools      research_question, read_links, compile_research, write_brief   (actions)
  prompt     research_workflow                                               (the recipe for using them)
  resources  config://settings, profiles://{name}, guideline://template       (read-only data)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

from pydantic import Field

from . import config, tools
from .profiles import PROFILE_NAMES, load_profiles

try:  # standalone FastMCP 2.x
    from fastmcp import FastMCP
except ImportError:  # FastMCP bundled with the official MCP SDK
    from mcp.server.fastmcp import FastMCP

SERVER_NAME = "market-research"
SERVER_VERSION = "0.2.0"

WorkingDir = Annotated[str, Field(description="Absolute path of the research folder. It holds guideline.md "
                                               "and receives memory/, research.md and output/.")]

WORKFLOW = """You are running a market-research workflow in the folder {working_dir}.
The tools do the work; you decide what to research next.

1. Read {working_dir}/guideline.md. If it is missing, ask the user for the topic, the angle and the
   key questions, write guideline.md from guidelines/TEMPLATE.md, and STOP for the user to confirm it.
2. If the guideline lists seed links, call read_links(working_dir).
3. Call research_question(working_dir, question) once per key question in the guideline.
4. Look at the data_gaps each call returns. For gaps that matter for the guideline's angle, call
   research_question again with a narrower question (company names, country, year). Stop after
   {max_rounds} follow-up calls, or earlier when the remaining gaps cannot be filled by web search.
5. Call compile_research(working_dir, remaining_gaps=[...]) with the gaps that are still open.
6. STOP and tell the user research.md is ready, with the number of findings and the open gaps.
   Continue only when they confirm (they may edit research.md or guideline.md first).
7. Call write_brief(working_dir). Report the final brief path, the evaluation verdict and failed checks.
   All versions are kept in output/, so the user can pick an earlier one if they prefer it.

Rules: never invent facts that are not in the tool results; keep the question list short and specific.
"""

def research_question(
    working_dir: WorkingDir,
    question: Annotated[str, Field(description="One specific web search question, under 12 words, with "
                                               "company names and country, e.g. 'Foodmandu market share Nepal 2024'.")],
) -> dict:
    """Search the web for ONE question, drop irrelevant pages, and extract sourced findings.

    Saves the result to memory/ and returns the findings and the data gaps still open, so you can decide
    whether another, narrower question is needed.
    """
    return tools.research_question(working_dir, question)


def read_links(
    working_dir: WorkingDir,
    urls: Annotated[Optional[list[str]], Field(description="Pages to read. Default: the seed links in guideline.md.")] = None,
) -> dict:
    """Read specific web pages (the user's trusted links) and extract sourced findings from them."""
    return tools.read_links(working_dir, urls)


def compile_research(
    working_dir: WorkingDir,
    remaining_gaps: Annotated[Optional[list[str]], Field(
        description="What is still unknown after all research rounds. Default: the gaps of the last round.")] = None,
) -> dict:
    """Merge all research rounds in memory/ into research.md and research.json (deduplicated)."""
    return tools.compile_research(working_dir, remaining_gaps)


def write_brief(
    working_dir: WorkingDir,
    iterations: Annotated[int, Field(ge=1, le=config.MAX_REVIEW_ITERATIONS,
                                     description="Maximum review/edit rounds.")] = config.REVIEW_ITERATIONS,
    evaluate: Annotated[bool, Field(description="Score retrieval, grounding and citations afterwards.")] = True,
) -> dict:
    """Write the brief from guideline.md and research.json.

    The Analyst drafts it using the writing profiles; the Critic reviews it in a separate context against
    the guideline, the research and the profiles; the Analyst applies the reviews. Every version is saved
    to output/ (draft_v0.md ... final.md, reviews.md).
    """
    return tools.write_brief(working_dir, iterations, evaluate)


def research_workflow(working_dir: str) -> str:
    """Step-by-step recipe for researching a market and writing the brief with this server's tools."""
    return WORKFLOW.format(working_dir=working_dir, max_rounds=config.MAX_GAP_ROUNDS * config.MAX_GAP_QUERIES)


def settings() -> str:
    """Server version, models, limits and thresholds."""
    return json.dumps({
        "server": SERVER_NAME, "version": SERVER_VERSION,
        "model": config.model_name(), "judge_model": config.judge_model_name(),
        "web_search": bool(config.tavily_api_key()), "tracing": config.opik_enabled(),
        "review_iterations": config.REVIEW_ITERATIONS, "max_sources": config.MAX_TOTAL_SOURCES,
        "profiles": list(load_profiles()), "banned_phrases": list(config.BANNED_PHRASES),
    }, indent=2)


def profile(name: str) -> str:
    """One writing profile: structure, terminology or character."""
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile {name!r}; choose one of {PROFILE_NAMES}")
    return load_profiles().get(name, "")


def guideline_template() -> str:
    """The guideline template to fill in for a new topic."""
    return (Path(config.PROJECT_ROOT) / "guidelines" / "TEMPLATE.md").read_text(encoding="utf-8")


def build_server() -> FastMCP:
    """Register tools, prompt and resources. Functions stay plain so they can be called and tested directly."""
    server = FastMCP(SERVER_NAME)
    for fn in (research_question, read_links, compile_research, write_brief):
        server.tool()(fn)
    server.prompt()(research_workflow)
    server.resource("config://settings")(settings)
    server.resource("profiles://{name}")(profile)
    server.resource("guideline://template")(guideline_template)
    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
