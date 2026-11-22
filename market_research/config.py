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
FINDING_SNIPPET_CHARS = 200
MIN_SNIPPET_CHARS = 100
SEARCH_QUERY_TEMPLATES = ("{q} market overview", "{q} competitors", "{q} latest news")
MAX_PLANNED_QUERIES = 3     # search queries planned from the user query
MAX_FINDINGS = 10           # numbered findings F1..F10 handed to the Analyst
THIN_EVIDENCE_FINDINGS = 4  # fewer findings than this: the brief must say the evidence is thin
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

# --- writing: profiles and few-shot examples (markdown files), evaluator-optimizer loop
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.getenv("PROFILES_DIR", os.path.join(PROJECT_ROOT, "profiles"))
MAX_FEW_SHOT = 1            # example briefs sent to the Analyst; raise only if quality drops without them
REVIEW_ITERATIONS = 1       # review -> edit rounds; the assignment allows ONE revision
MAX_REVIEW_ITERATIONS = 1   # hard cap, also applied in orchestrator()
MAX_SEED_LINKS = 5          # guideline links read before searching

# --- judge calibration (labelled dataset of real runs)
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "eval_data")
SPLIT_SHARES = (("train", 0.4), ("dev", 0.3), ("test", 0.3))
MAX_JUDGE_EXAMPLES = 4      # train examples shown to the calibrated judge

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


def use_sample_env() -> bool:
    """USE_SAMPLE_DATA=1 forces labelled sample data instead of web search."""
    return os.getenv("USE_SAMPLE_DATA", "").strip().lower() in ("1", "true", "yes", "on")


def model_name() -> str:
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def judge_model_name() -> str:
    return os.getenv("GROQ_JUDGE_MODEL") or model_name()


OPIK_PROJECT = os.getenv("OPIK_PROJECT_NAME", "market-research")


def opik_enabled() -> bool:
    """Tracing is opt-in: set OPIK_API_KEY (Comet cloud) or OPIK_URL_OVERRIDE (self-hosted)."""
    return bool(_key("OPIK_API_KEY") or os.getenv("OPIK_URL_OVERRIDE"))
