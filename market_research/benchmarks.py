"""Cross-run benchmark log: one JSON line per run in runs/benchmarks.jsonl, so runs can be compared over time.

Each record holds status, total time and the ordered per-step metrics (seconds, LLM calls, tokens) that the
workflow already measures for every node. Logging never breaks a run: write errors are swallowed.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config
from .models import WorkflowState

FILE_NAME = "benchmarks.jsonl"


def benchmark_record(state: WorkflowState, started_at: datetime, seconds: float) -> dict[str, Any]:
    metrics = [m.model_dump() for m in state.metrics]
    return {
        "run_id": uuid.uuid4().hex,
        "query": state.query,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "error" if state.error else "ok",
        "error": state.error,
        "total_seconds": round(seconds, 2),
        "model": config.model_name(),
        "data_mode": state.research.mode if state.research else None,
        "findings": len(state.research.findings) if state.research else 0,
        "versions": len(state.drafts),
        "approved": state.critique.approved if state.critique else None,
        "verdict": state.evaluation.verdict if state.evaluation else None,
        "grounding_flags": len(state.grounding),
        "llm_calls": sum(m["llm_calls"] for m in metrics),
        "input_tokens": sum(m["input_tokens"] for m in metrics),
        "output_tokens": sum(m["output_tokens"] for m in metrics),
        "steps": metrics,  # in the order the nodes ran
        "run_dir": state.run_dir,
    }


def record_run(state: WorkflowState, started_at: datetime, seconds: float, base_dir: str) -> str | None:
    """Append one record to <base_dir>/benchmarks.jsonl. Returns the file path, or None if it could not be written."""
    path = Path(base_dir) / FILE_NAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(benchmark_record(state, started_at, seconds), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        return None
    return str(path)


def load_benchmarks(base_dir: str = config.RUNS_DIR) -> list[dict[str, Any]]:
    """All records, oldest first. Lines that are not valid JSON (e.g. a half-written line) are skipped."""
    path = Path(base_dir) / FILE_NAME
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
