# Multi-Agent Market Research App

A small Streamlit app that takes a company or industry (for example
*"Analyze the food delivery market in Nepal"*) and produces a short research brief using
four roles that run in sequence. It also includes an **evaluation system** that tells you whether the
retrieval-augmented pipeline (live web search -> grounded brief) is actually working.

## How it works

```
guideline.md (optional) ─┐
                         v
Orchestrator -> Researcher -> Analyst (v0) -> [Critic -> Analyst edits] x N -> Evaluator
                                  ^                 ^
              profiles/*.md + few-shot example      guideline > research > profiles
```

The research part is exploratory (it searches again for what is missing); the writing part is a fixed
workflow held in place by a guideline, writing profiles and a reviewer. That split follows the
"deep research agent + writing workflow" design from the Towards AI workshop
*Build Your Own Deep Research Agents*, adapted to market research on Groq + Tavily.

| Role | File | What it does |
|---|---|---|
| Orchestrator | `market_research/workflow.py` | Receives the query, runs the roles in order, allows at most one revision |
| Researcher | `market_research/agents/researcher.py`, `search.py` | Searches the web with Tavily (or uses clearly labelled sample data) and extracts findings with source URLs and data gaps |
| Analyst | `market_research/agents/analyst.py` | Writes the brief from the findings, the guideline and the writing profiles, then applies reviews |
| Critic | `market_research/agents/critic.py` | Reviews each version in its own context against the guideline, the findings and the profiles |

All roles use the same Groq model (`GROQ_MODEL`). Every handoff goes through Pydantic models in
`market_research/models.py` (`Research`, `Brief`, `Critique`, all held in one `WorkflowState`); model output is validated
against these schemas, with one automatic retry on invalid JSON.

**Research loop.** After the first extraction, the Researcher reads its own `data_gaps` (for example "market
size not found"), plans targeted queries for them, searches again, filters the new documents and re-extracts. It stops
after `MAX_GAP_ROUNDS` (2), when nothing new turns up, or at `MAX_TOTAL_SOURCES` (12) documents, to protect the
context budget. If a gap round fails (for example a rate limit), the earlier findings are kept.

**Guideline (what to write).** Instead of a one-line query you can pass a markdown guideline
(`guidelines/TEMPLATE.md`, example: `guidelines/nepal-food-delivery.md`) with the angle, audience, key questions,
must-include items, seed links and constraints. Key questions become the first searches, seed links are read
before searching (Tavily Extract), and the Analyst and Critic both see the guideline. Its constraints override the
profiles. A specific guideline improves the brief more than any other setting.

**Profiles and few-shot example (how to write).** `profiles/structure.md`, `terminology.md` and `character.md`
are plain markdown you can edit; the banned-phrase list from `config.py` is appended to the terminology profile.
`profiles/examples/` holds example briefs (fictional facts, style only); only `MAX_FEW_SHOT` (1) is sent, to keep
the prompt small. Add examples only if quality drops without them.

**Review loop (evaluator-optimizer).** The Critic reviews each version in a separate LLM call and returns typed
issues (`rule`, `section`, `category`, `comment`). `rule` says which source of truth is broken, and the Analyst
fixes them in priority order: guideline, then research, then the structure/terminology/character profiles
(then competitor limit > unsupported claim > style within a rule). The loop runs a fixed budget of
`REVIEW_ITERATIONS` rounds (2; `--iterations N` or the sidebar slider, max 5) and stops early when the Critic
approves. There is no score threshold: writing quality is subjective and such thresholds make the loop noisy.
Every version is kept (`draft_v0.md` ... `draft_vN.md`), since an earlier one is sometimes better.

**Run files.** Every run writes `runs/<time>-<query>/` with `research.md`, `draft_v0.md` ... `draft_vN.md`,
`reviews.md`, `final.md`, `guideline.md` (if given) and `state.json` (full shared state, including every review and
per-step seconds and LLM tokens). The UI shows the same numbers under "Research rounds
and run cost". Use `--no-save` on the CLI to skip writing files.

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
python -m market_research run --guideline guidelines/nepal-food-delivery.md --iterations 3
# options: --sample  --no-eval  --force-revision (tests the revision path)  --iterations N
```

### Claude Code: MCP server and skill

The same tools are served over MCP (`market_research/mcp_server.py`, FastMCP), so an agent harness such as
Claude Code can drive the research itself. Here the agent is the brain: it reads the gaps each search returns and
decides the next question. The server is the hands.

| Kind | Name | What it does |
|---|---|---|
| tool | `research_question(working_dir, question)` | One search round: search, filter, extract findings; saves `memory/NNN-search.json` |
| tool | `read_links(working_dir, urls?)` | Reads the guideline's seed links (or given URLs) into memory |
| tool | `compile_research(working_dir, remaining_gaps?)` | Merges memory into `research.md` / `research.json` |
| tool | `write_brief(working_dir, iterations, evaluate)` | Draft + review loop + evaluation; writes `output/` |
| prompt | `research_workflow(working_dir)` | The recipe: when to call which tool, and where to stop for you |
| resource | `config://settings`, `profiles://{name}`, `guideline://template` | Read-only settings, profiles and template |

All handoffs are files in the working folder (`guideline.md`, `memory/`, `research.md`, `output/`), so you can read
or edit any step before the next one.

Setup: `pip install fastmcp` (already in `requirements.txt`), then start Claude Code from the project root with the
virtualenv activated. `.mcp.json` registers the server (`/mcp` should list `market-research`; if Claude Code does
not use your venv, change `"command"` to the full path of `venv/bin/python`). The project skill
`.claude/skills/market-research/SKILL.md` wraps the workflow: ask "research the food delivery market in Nepal" or
`/market-research`, and Claude creates `research/<slug>/guideline.md`, stops for you to confirm it, researches,
stops again at `research.md`, and writes the brief. Skills load only their name and description until used, which
keeps the context small; the MCP server is what makes the logic easy to share with other clients.

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
| `style_check` | rule | No AI-slop phrases, em dashes or vague generalisations like "most companies"? (informational) |
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

### Calibrating an expert judge (pass/fail + critique)

The metrics above check grounding. Whether a brief is actually *good* needs your judgement, so the project
includes an LLM judge you align with your own labels, treated like a binary classifier:

```bash
# 1. Label real outputs of the system (never let an LLM write the outputs or labels).
#    pass/fail plus 2-3 sentences; for a fail, stop at the first real problem.
python -m market_research label runs/<run> --label fail --critique "States a 2023 market size the findings do not contain."
#    Rows go to eval_data/dataset.jsonl and get a fixed train/dev/test split (40/30/30 by id hash, or --split).

# 2. Measure the judge on dev. Train rows are its few-shot examples (balanced pass/fail, MAX_JUDGE_EXAMPLES=4).
python -m market_research calibrate --split dev   # prints F1/precision/recall on "fail" and every disagreement

# 3. Adjust JUDGE_SYSTEM in evaluation/calibration.py or add train rows, repeat; then check once on test.
python -m market_research calibrate --split test  # dev much higher than test = the judge overfits

# 4. Use it on new runs.
python -m market_research judge runs/<run>
```

Aim for at least 20-30 labelled runs (more is better) before trusting the score; with five per split a perfect F1
means little. Results are saved to `eval_data/calibration-<split>.json`.

### Tracing (optional)

`pip install opik` and set `OPIK_API_KEY` (Comet cloud, plus `OPIK_WORKSPACE`) or `OPIK_URL_OVERRIDE`
(self-hosted). Each workflow step, LLM call and MCP tool call then becomes a span in the `market-research`
project (`OPIK_PROJECT_NAME`), with inputs, outputs and latency. Without it nothing changes; the per-step time and
token counts are in `state.json` either way.

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
  artifacts.py               Saves research.md, draft_v0.md, final.md, state.json per run
  rendering.py               Markdown rendering of briefs and evaluation reports
  urls.py                    URL normalisation and repair (snap_urls)
  search.py                  Tavily retrieval
  llm.py                     Groq wrapper with JSON validation, retry and rate-limit backoff
  workflow.py                Orchestrator (LangGraph, review loop)
  guideline.py               Guideline markdown parser
  profiles.py                Loads writing profiles and few-shot examples
  tools.py                   File-based research/writing tools (used by the MCP server)
  mcp_server.py              FastMCP server: tools, prompt, resources
  tracing.py                 Optional Opik tracing
  agents/                    researcher.py, analyst.py, critic.py, sample_data.py
  evaluation/                rules.py (deterministic), judges.py (LLM judges),
                             verdict.py (evaluate), tamper.py, batch.py,
                             calibration.py (labelled dataset, expert judge, F1)
guidelines/                  TEMPLATE.md and an example guideline
profiles/                    structure.md, terminology.md, character.md, examples/
.mcp.json                    Registers the MCP server with Claude Code
.claude/skills/market-research/SKILL.md   Claude Code skill for the end-to-end workflow
eval_data/                   dataset.jsonl (your labels), calibration results
  cli.py / __main__.py       `python -m market_research {run,eval}`
tests/                       pytest tests (no API keys required)
test_queries.json            Queries for the batch evaluation
.env.example                 Placeholder environment variables
sample_output.md             Example brief with evaluation report
```

## Limitations

- Quality depends on what web search returns; the brief lists missing information instead of guessing.
- The review loop has a fixed budget; the last edit is not reviewed again (look at `reviews.md` and earlier versions).
- The expert judge is only as good as your labels; with few labels its F1 is not meaningful.
- YouTube transcripts and GitHub ingestion from the workshop are not included (they need Gemini / gitingest);
  seed links cover web pages.
- LLM-judge metrics are themselves model outputs: treat them as a guide, not ground truth.
- Groq's free tier has rate limits (per minute and about 200k tokens per day per model). Gap rounds add one or two extraction calls, and evaluation adds more; use `--no-eval` or disable gap rounds (`MAX_GAP_ROUNDS = 0` in `config.py`) when quota is tight.
- No caching or deployment config (kept simple on purpose).

## Credits

Structure and ideas were informed by open-source market-research agent projects (MIT licensed), including
Market-Intelligence-Agent and Market-Research-Agent. No code was copied from them.
# Research_Agent
