"""Batch evaluation over a queries file, including the tamper negative control."""
from __future__ import annotations

import json

from ..models import MetricResult
from ..workflow import orchestrator
from .judges import brief_groundedness
from .rules import citation_validity
from .tamper import tamper


def run_batch(queries_file: str, use_sample: bool = False) -> list[dict]:
    with open(queries_file, encoding="utf-8") as f:
        queries = json.load(f)
    rows = []
    for q in queries:
        print(f"\n=== {q}")
        try:
            state = orchestrator(q, use_sample=use_sample, evaluate_rag=True)
            t = tamper(state.final or state.draft)
            tg = brief_groundedness(state, t)
            tc = citation_validity(state, t)
            m = {x.name: x for x in state.evaluation.metrics}
            real_g = m["brief_groundedness"].score
            rows.append({
                "query": q, "verdict": state.evaluation.verdict,
                "relevance": m.get("retrieval_relevance", MetricResult(name="x", score=0, passed=False)).score,
                "groundedness": real_g, "citation": m["citation_validity"].score,
                "tampered_groundedness": tg.score, "tampered_citation": tc.score,
                "evaluator_ok": (not tc.passed) and tg.score < real_g,
            })
        except Exception as e:
            rows.append({"query": q, "error": str(e)})
        print(rows[-1])
    return rows


def print_table(rows: list[dict]) -> None:
    print("\n{:<45} {:<16} {:>5} {:>5} {:>5} {:>7} {:>7} {}".format(
        "query", "verdict", "rel", "grnd", "cite", "t.grnd", "t.cite", "evaluator OK?"))
    for r in rows:
        if "error" in r:
            print(f"{r['query'][:44]:<45} ERROR: {r['error'][:60]}")
        else:
            print("{:<45} {:<16} {:>5.2f} {:>5.2f} {:>5.2f} {:>7.2f} {:>7.2f} {}".format(
                r["query"][:44], r["verdict"], r["relevance"], r["groundedness"], r["citation"],
                r["tampered_groundedness"], r["tampered_citation"], "yes" if r["evaluator_ok"] else "NO"))
