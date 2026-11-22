"""Features borrowed from the CrewAI reference project: evidence rules, cross-run benchmark log, PDF export."""
import importlib
import json
import re
from datetime import UTC, datetime

import pytest

from market_research import benchmarks, config, pdf
from market_research.models import Brief, Finding, Research, StepMetric, WorkflowState

# `market_research.agents` re-exports functions with the same names as its submodules.
analyst_mod = importlib.import_module("market_research.agents.analyst")
critic_mod = importlib.import_module("market_research.agents.critic")
researcher_mod = importlib.import_module("market_research.agents.researcher")


def test_evidence_rules_are_in_every_role_prompt():
    assert config.EVIDENCE_RULE_RESEARCH in researcher_mod.SYSTEM
    assert config.EVIDENCE_RULE_WRITE in analyst_mod.SYSTEM
    assert config.EVIDENCE_RULE_REVIEW in critic_mod.SYSTEM
    assert "Could not be verified" in analyst_mod.SYSTEM and "forecast" in critic_mod.SYSTEM


def _state(**kw) -> WorkflowState:
    research = Research(mode="web", focus=["Nepal"], findings=[
        Finding(id="F1", claim="Foodmandu had 1M users in 2024.", source_title="Foodmandu",
                source_url="https://foodmandu.com/a")])
    brief = Brief(summary="Nepal food delivery is growing [F1].", key_observations=["Users: 1M (2024) [F1]"],
                  missing_information=["Could not be verified: market size"])
    return WorkflowState(query="Nepal food delivery", research=research, draft=brief, final=brief,
                         metrics=[StepMetric(step="researcher", seconds=1.5, llm_calls=2, input_tokens=100,
                                             output_tokens=40),
                                  StepMetric(step="analyst", seconds=2.0, llm_calls=1, input_tokens=50,
                                             output_tokens=60)], **kw)


def test_benchmark_log_appends_one_line_per_run(tmp_path):
    t = datetime.now(UTC)
    benchmarks.record_run(_state(), t, 3.5, str(tmp_path))
    benchmarks.record_run(_state(error="The LLM rate limit was hit."), t, 0.4, str(tmp_path))
    lines = (tmp_path / "benchmarks.jsonl").read_text().splitlines()
    assert len(lines) == 2
    ok, failed = (json.loads(line) for line in lines)
    assert ok["status"] == "ok" and ok["total_seconds"] == 3.5 and ok["llm_calls"] == 3
    assert ok["input_tokens"] == 150 and [s["step"] for s in ok["steps"]] == ["researcher", "analyst"]
    assert failed["status"] == "error" and "rate limit" in failed["error"]
    assert ok["run_id"] != failed["run_id"]


def test_load_benchmarks_skips_broken_lines(tmp_path):
    (tmp_path / "benchmarks.jsonl").write_text('{"query": "a"}\n{half a line\n{"query": "b"}\n')
    assert [r["query"] for r in benchmarks.load_benchmarks(str(tmp_path))] == ["a", "b"]
    assert benchmarks.load_benchmarks(str(tmp_path / "missing")) == []


def test_benchmark_write_failure_never_breaks_a_run(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")  # a file where the directory should be
    assert benchmarks.record_run(_state(), datetime.now(UTC), 1.0, str(blocker)) is None


def test_failed_run_is_benchmarked(monkeypatch, tmp_path):
    from market_research import workflow
    from market_research.llm import LLMError

    def boom(*_args, **_kwargs):
        raise LLMError("The LLM rate limit was hit on every Groq key.")

    monkeypatch.setattr(workflow, "researcher", boom)
    st = workflow.orchestrator("Nepal food delivery", use_sample=True, save_dir=str(tmp_path), raise_errors=False)
    assert st.error
    [rec] = benchmarks.load_benchmarks(str(tmp_path))
    assert rec["status"] == "error" and rec["query"] == "Nepal food delivery"


def test_inline_markup_links_escapes_and_emphasis():
    out = pdf.inline_markup(r"**Bold** & [\[F4 · foodmandu.com\]](https://foodmandu.com/a?x=1&y=2) *note*")
    assert "<b>Bold</b>" in out and "&amp;" in out and "<i>note</i>" in out
    assert '<link href="https://foodmandu.com/a?x=1&amp;y=2"' in out and "[F4 · foodmandu.com]" in out
    assert "<script>" not in pdf.inline_markup("<script>alert(1)</script>")


def test_llm_unicode_punctuation_survives_the_built_in_fonts():
    # non-breaking hyphen, narrow no-break space, en dash, minus: seen in real gpt-oss output
    out = pdf.inline_markup("pandemic‑induced Series B 14–15% −2%")
    assert out == "pandemic-induced Series B 14–15% -2%" and "?" not in out


def test_brief_pdf_is_a_valid_pdf():
    pytest.importorskip("reportlab")
    data = pdf.brief_to_pdf(_state())
    assert data.startswith(b"%PDF") and len(data) > 1000


def _pages(data: bytes) -> list[str]:
    """Decoded text-drawing operators of each page, in order (no PDF library needed).

    ReportLab writes each page's content as an ASCII85 + Flate stream; page streams are the ones with text.
    """
    import base64
    import zlib

    out = []
    for raw in re.findall(rb"stream\r?\n(.*?)endstream", data, re.S):
        body = raw.strip()
        try:
            if body.endswith(b"~>"):
                body = base64.a85decode(body[:-2].replace(b"\n", b"").replace(b"\r", b""))
            text = zlib.decompress(body).decode("latin-1")
        except (ValueError, zlib.error):
            continue
        if " Tj" in text or "TJ" in text:
            out.append(text)
    return out


A4_MEDIABOX = re.compile(rb"/MediaBox \[\s*0 0 595\.27\d* 841\.88\d*\s*\]")


def test_markdown_pdf_handles_tables_quotes_rules_and_numbered_lists():
    pytest.importorskip("reportlab")
    md = ("# T\n\n> **Warning**\n\n## Eval\n| A | B |\n|---|---|\n| 1 | long reason text |\n\n---\n*done*\n\n"
          "1. **Supply:** first point\n2) second point")
    pages = _pages(pdf.markdown_to_pdf(md, "Title here"))
    text = "".join(pages)
    for needle in ("Title here", "Warning", "Eval", "long reason text", "1.", "first point", "2.", "second point",
                   "MARKET RESEARCH BRIEF", "Page 1 of 1", "Generated "):
        assert needle in text, needle


def test_pdf_pages_are_a4_and_numbered_on_every_page():
    pytest.importorskip("reportlab")
    long_md = "\n\n".join(f"## Section {i}\n" + "Market detail sentence with figures. " * 40 for i in range(12))
    data = pdf.markdown_to_pdf(long_md, "Long report")
    pages = _pages(data)
    total = len(pages)
    assert total > 1
    assert len(A4_MEDIABOX.findall(data)) == total  # every page is A4 (595.28 x 841.89 pt)
    for n, page in enumerate(pages, 1):
        assert f"Page {n} of {total}" in page


def test_empty_brief_still_gives_a_one_page_pdf():
    pytest.importorskip("reportlab")
    data = pdf.brief_to_pdf(WorkflowState(query="Empty report test"))
    pages = _pages(data)
    assert len(pages) == 1 and "Empty report test" in pages[0] and "Page 1 of 1" in pages[0]
