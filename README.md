# Multi-Agent Market Research App

A small Streamlit app that takes a company or industry (for example
*"Analyze the food delivery market in Nepal"*) and produces a short research brief using
four roles that run in sequence. It also includes an **evaluation system** that tells you whether the
retrieval-augmented pipeline (live web search -> grounded brief) is actually working.

## How it works

```
Orchestrator -> Researcher -> Analyst -> Critic -> (Analyst revises once, if needed) -> Evaluator
```

| Role | File | What it does |
|---|---|---|
| Orchestrator | `market_research/workflow.py` | Receives the query, runs the roles in order, allows at most one revision |
| Researcher | `market_research/agents/researcher.py`, `search.py` | Searches the web with Tavily (or uses clearly labelled sample data) and extracts findings with source URLs and data gaps |
| Analyst | `market_research/agents/analyst.py` | Writes the brief using only the findings |
| Critic | `market_research/agents/critic.py` | Checks the brief against the findings and approves it or requests one revision |

All roles use the same Groq model (`GROQ_MODEL`). Every handoff goes through Pydantic models in
`market_research/models.py` (`Research`, `Brief`, `Critique`, all held in one `WorkflowState`); model output is validated
against these schemas, with one automatic retry on invalid JSON.

**Brief contents:** summary, up to two competitors (enforced in code), key observations, sources, missing information.

## Setup

Requires Python 3.10+, a [Groq API key](https://console.groq.com/keys) and (recommended) a
[Tavily API key](https://app.tavily.com). Both have free tiers.

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # Windows: copy .env.example .env
# open .env and replace the placeholders with your real keys
```

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501, type a company or industry and click **Start research**. Progress for each
role appears live. The page shows the brief, the evaluation scorecard, and an "Agent handoffs" expander
with the critic's verdict and the full shared state. The brief can be downloaded as Markdown.

**No Tavily key?** The Researcher falls back to fixed, clearly labelled **sample data** (about Nepal food
delivery) and the app shows a warning banner. You can also force this with the sidebar checkbox. A Groq key
is always required.

### Command line

```bash
python -m market_research run "Analyze the food delivery market in Nepal" --out sample_output.md
# options: --sample  --no-eval  --force-revision (tests the one-revision path)
```

## Evaluating the RAG ("is it working?")

After each run the evaluator scores both halves of the pipeline:

| Metric | Type | Question |
|---|---|---|
| `retrieval_check` | rule | Were >=3 sources from >=2 domains retrieved live, with non-empty snippets? |
| `retrieval_relevance` | LLM judge | What share of the retrieved documents is relevant to the query? (>= 0.6) |
| `research_faithfulness` | LLM judge | Are the Researcher's findings supported by the retrieved snippets? (>= 0.7) |
| `brief_groundedness` | LLM judge | Are the brief's claims supported by the findings and snippets? (>= 0.7) |
| `citation_validity` | rule | Was every URL cited in the brief actually retrieved? (must be 100%) |
| `structure_check` | rule | Summary, <=2 competitors, >=3 observations, sources, missing info present? |
| `gap_honesty` | LLM judge | Does the brief admit what the evidence does not cover? (informational) |

**Verdict:** `RAG WORKING` when all required checks pass, `RAG NOT WORKING` (with the failing checks listed)
otherwise, and `RAG NOT ACTIVE` when sample data was used, because retrieval was not exercised.
Thresholds live in `market_research/config.py`. Judge calls use temperature 0 and
`GROQ_JUDGE_MODEL`; set it to a different model than `GROQ_MODEL` so the model is not grading itself.

**Does the evaluator itself work?** Run the batch evaluation:

```bash
python -m market_research eval   # runs test_queries.json, writes eval_results.json
```

For every query it also scores a *tampered* copy of the brief (one invented claim and one fake URL) and
checks that groundedness drops and citation validity fails. The last column, "evaluator OK?", should be `yes`.

Note: this is retrieval-augmented generation over live web search results; there is no vector database.

## Tests

```bash
pytest
```

The tests need no API keys: they cover the rule-based metrics and run the whole workflow with the LLM and
search mocked (including the one-revision cap and the tamper check).

## Project structure

```
app.py                       Streamlit UI (thin entry point, one render function per section)
market_research/
  config.py                  Env lookups, limits, evaluation thresholds
  models.py                  Pydantic models for shared data
  rendering.py               Markdown rendering of briefs and evaluation reports
  urls.py                    URL normalisation and repair (snap_urls)
  search.py                  Tavily retrieval
  llm.py                     Groq wrapper with JSON validation, retry and rate-limit backoff
  workflow.py                Orchestrator
  agents/                    researcher.py, analyst.py, critic.py, sample_data.py
  evaluation/                rules.py (deterministic), judges.py (LLM judges),
                             verdict.py (evaluate), tamper.py, batch.py
  cli.py / __main__.py       `python -m market_research {run,eval}`
tests/                       pytest tests (no API keys required)
test_queries.json            Queries for the batch evaluation
.env.example                 Placeholder environment variables
sample_output.md             Example brief with evaluation report
```

## Limitations

- Quality depends on what web search returns; the brief lists missing information instead of guessing.
- The Critic gets one revision cycle only, and the revised brief is not reviewed again.
- LLM-judge metrics are themselves model outputs: treat them as a guide, not ground truth.
- Groq's free tier has rate limits; the app retries with backoff but a very busy minute can still fail.
- No caching or deployment config (kept simple on purpose).

## Credits

Structure and ideas were informed by open-source market-research agent projects (MIT licensed), including
Market-Intelligence-Agent and Market-Research-Agent. No code was copied from them.
# Research_Agent
