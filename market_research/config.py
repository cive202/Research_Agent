"""Central configuration: environment lookups, limits and evaluation thresholds."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# --- workflow
MAX_QUERY_CHARS = 200

# --- search
MAX_SOURCES = 8
SNIPPET_CHARS = 600
MIN_SNIPPET_CHARS = 100
SEARCH_QUERY_TEMPLATES = ("{q} market overview", "{q} competitors", "{q} latest news")
MAX_PLANNED_QUERIES = 5
RESULTS_PER_QUERY = 4
URL_SNAP_CUTOFF = 0.9

# --- relevance filtering: re-search once if fewer than this share (or count) of documents survive
MIN_KEPT_SHARE = 0.6
MIN_KEPT_SOURCES = 4

# --- gap-driven research: after extraction, search again for what is still missing
MAX_GAP_ROUNDS = 2          # extra search rounds that target data_gaps (0 disables the loop)
MAX_GAP_QUERIES = 3         # queries per gap round
MAX_TOTAL_SOURCES = 12      # context budget: never feed more documents than this to the Researcher

# --- run artifacts
RUNS_DIR = "runs"

# --- style: phrases the Analyst is told to avoid and the style_check metric flags
BANNED_PHRASES = (
    "delve", "tapestry", "vibrant", "intricacies", "rapidly evolving", "ever-evolving", "fast-paced",
    "game-changer", "game changer", "game-changing", "unlock", "in today's", "it's not about",
    "it is not about", "navigate the complexities", "testament to",
)
VAGUE_GENERALIZATION = r"\b(most|many|numerous|countless)\s+(companies|people|players|users|businesses|teams|consumers|firms)\b"

# --- LLM
LLM_TIMEOUT = 60.0
LLM_MAX_TOKENS = 2500
LLM_ATTEMPTS = 3

# --- evaluation thresholds
MIN_SOURCES = 3
MIN_DOMAINS = 2
MIN_RELEVANCE = 0.6
MIN_FAITHFULNESS = 0.7
MIN_GROUNDEDNESS = 0.7


def _key(name: str) -> str | None:
    value = os.getenv(name)
    return None if not value or value.startswith("your-") else value


def groq_api_key() -> str | None:
    return _key("GROQ_API_KEY")


def tavily_api_key() -> str | None:
    return _key("TAVILY_API_KEY")


def model_name() -> str:
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def judge_model_name() -> str:
    return os.getenv("GROQ_JUDGE_MODEL") or model_name()
