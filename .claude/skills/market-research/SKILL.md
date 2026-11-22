---
name: market-research
description: Research a company or market and write a short, cited market-research brief, using the market-research MCP tools (research_question, read_links, compile_research, write_brief). Use when the user asks to research, analyse or write a brief about a company, an industry or a market, or points at a folder with a guideline.md.
---

# Market research brief

The `market-research` MCP server does the searching and writing. You are the planner: you choose the
questions, read the gaps each round reports and decide when the research is good enough.

## 1. Set up the working folder

- If the user gives a folder with `guideline.md`, use it (absolute path).
- Otherwise create `research/<short-slug>/` in the project root, copy `guidelines/TEMPLATE.md` to
  `guideline.md` there and fill it in from what the user said: topic (title), angle, audience,
  key questions (3-5 specific search queries with company names and country), must-include items,
  seed links, constraints.
- **Stop and show the guideline to the user.** A specific guideline matters more than anything else
  for the quality of the brief. Continue when they confirm.

## 2. Research

1. If the guideline has seed links, call `read_links(working_dir)`.
2. Call `research_question(working_dir, question)` for each key question.
3. Read the `data_gaps` that come back. For gaps that matter to the angle, ask a narrower follow-up
   question (names, country, year). At most 6 follow-up calls. Skip gaps no web search can fill
   (private company revenue, for example).
4. Call `compile_research(working_dir, remaining_gaps=[...])` with the gaps still open.
5. **Stop**: tell the user `research.md` is ready (number of findings, open gaps). They may edit it.

## 3. Write

Call `write_brief(working_dir)`. Then report:
- the path of `output/final.md` and how many versions were written (`output/draft_v*.md`),
- the evaluation verdict and any failed checks,
- that earlier versions are kept if they prefer one of them.

Never add facts that are not in the tool results. If a tool returns `status: error`, report the error
(usually a missing `GROQ_API_KEY` or `TAVILY_API_KEY` in `.env`) instead of working around it.
