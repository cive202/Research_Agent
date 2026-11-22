"""An expert-aligned LLM judge, calibrated like a binary classifier.

1. Build the dataset: run the real system, read the output, and label it yourself
   (`python -m market_research label runs/<run> --label fail --critique "..."`). Label pass/fail and write
   2-3 sentences on why; for a fail, stop at the first real problem. Each row is assigned to train/dev/test.
2. The judge sees the guideline, findings and brief, and uses the *train* rows as few-shot examples.
3. Calibrate: `python -m market_research calibrate --split dev`, read the disagreements, adjust the prompt
   or add train examples, repeat. Then run once on `test`. A dev score far above the test score means
   the judge overfits the examples.

"fail" is the positive class: the judge's job is to catch bad briefs, so recall on fails matters most.
Labels must come from a person; never let an LLM write the labels or the outputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .. import config
from ..llm import ask_json
from ..models import Brief, MetricResult, WorkflowState
from ..profiles import writing_context
from ..rendering import brief_to_markdown

Label = Literal["pass", "fail"]
Split = Literal["train", "dev", "test"]


class LabelRow(BaseModel):
    id: str
    query: str
    run_dir: str = Field(description="Run folder holding state.json (relative to the project root if inside it).")
    version: Optional[int] = Field(default=None, description="Draft version labelled; None = the final brief.")
    label: Label
    critique: str
    split: Split


class Verdict(BaseModel):
    critique: str = Field(description="2-3 sentences: why it passes, or the FIRST real problem that makes it fail.")
    label: Label = Field(description="pass or fail.")


JUDGE_SYSTEM = (
    "You are a senior market analyst deciding whether a research brief is good enough to send to a client. "
    "A brief FAILS if any of these is true: it states a fact or figure the findings do not support; it ignores "
    "the guideline's angle or a key question the findings could answer; it hides an important gap; it lists "
    "more than two competitors; it is vague (e.g. 'most companies') or reads like generic AI filler. "
    "Otherwise it PASSES. Judge the way the labelled examples do. Write the critique first, stopping at the "
    "first real problem, then the label."
)


# ---------------------------------------------------------------- dataset

def dataset_path() -> Path:
    return Path(config.EVAL_DATA_DIR) / "dataset.jsonl"


def load_dataset(path: Optional[Path] = None) -> list[LabelRow]:
    p = path or dataset_path()
    if not p.is_file():
        return []
    return [LabelRow.model_validate_json(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def assign_split(row_id: str) -> Split:
    """Deterministic split from a hash of the id, so re-labelling never moves a row between splits."""
    x = int(hashlib.sha1(row_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, share in config.SPLIT_SHARES:
        acc += share
        if x < acc:
            return name  # type: ignore[return-value]
    return "test"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(config.PROJECT_ROOT)))
    except ValueError:
        return str(path.resolve())


def _abs(run_dir: str) -> Path:
    p = Path(run_dir)
    return p if p.is_absolute() else Path(config.PROJECT_ROOT) / p


def load_state(run_dir: str) -> WorkflowState:
    f = _abs(run_dir) / "state.json"
    if not f.is_file():
        raise FileNotFoundError(f"No state.json in {run_dir}")
    return WorkflowState.model_validate_json(f.read_text(encoding="utf-8"))


def add_label(run_dir: str, label: Label, critique: str, split: Optional[Split] = None,
              version: Optional[int] = None, path: Optional[Path] = None) -> LabelRow:
    """Add or replace the label of one run (and version). Returns the stored row."""
    state = load_state(run_dir)
    if version is not None and not 0 <= version < len(state.drafts):
        raise ValueError(f"Run has versions 0..{len(state.drafts) - 1}")
    rid = Path(run_dir).name + (f"@v{version}" if version is not None else "")
    rows = [r for r in load_dataset(path) if r.id != rid]
    row = LabelRow(id=rid, query=state.query, run_dir=_rel(_abs(run_dir)), version=version, label=label,
                   critique=critique.strip(), split=split or assign_split(rid))
    rows.append(row)
    p = path or dataset_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    return row


# ---------------------------------------------------------------- judge

def _brief_of(state: WorkflowState, version: Optional[int]) -> Brief:
    if version is not None:
        return state.drafts[version]
    return state.final or state.draft


def _case(state: WorkflowState, brief: Brief) -> str:
    g = state.guideline.to_prompt() if state.guideline else f"Topic: {state.query}"
    r = state.research
    findings = "\n".join(f"- {f.claim}" for f in (r.findings if r else [])) or "- (none)"
    gaps = "\n".join(f"- {x}" for x in (r.data_gaps if r else [])) or "- (none)"
    return (f"GUIDELINE:\n{g}\n\nFINDINGS:\n{findings}\n\nDATA GAPS:\n{gaps}\n\n"
            f"BRIEF:\n{brief_to_markdown(state, brief)}")


def _examples(rows: list[LabelRow]) -> str:
    # Balance pass and fail examples so the judge does not learn a default answer.
    fails = [r for r in rows if r.label == "fail"]
    passes = [r for r in rows if r.label == "pass"]
    picked: list[LabelRow] = []
    while (fails or passes) and len(picked) < config.MAX_JUDGE_EXAMPLES:
        for bucket in (fails, passes):
            if bucket and len(picked) < config.MAX_JUDGE_EXAMPLES:
                picked.append(bucket.pop(0))
    parts = []
    for n, r in enumerate(picked, 1):
        try:
            st = load_state(r.run_dir)
        except FileNotFoundError:
            continue
        parts.append(f"### Example {n}\n{_case(st, _brief_of(st, r.version))}\n\n"
                     f"EXPERT CRITIQUE: {r.critique}\nEXPERT LABEL: {r.label}")
    return "\n\n".join(parts)


def judge(state: WorkflowState, brief: Optional[Brief] = None, train: Optional[list[LabelRow]] = None) -> Verdict:
    train = train if train is not None else [r for r in load_dataset() if r.split == "train"]
    ex = _examples(train)
    profiles = writing_context(include_examples=False)
    user = ((f"Writing profiles the brief should follow:\n{profiles}\n\n" if profiles else "")
            + (f"Labelled examples:\n\n{ex}\n\n" if ex else "")
            + "### Brief to judge\n" + _case(state, brief or _brief_of(state, None)))
    return ask_json(Verdict, JUDGE_SYSTEM, user, model=config.judge_model_name(), temperature=0.0)


def expert_judge_metric(state: WorkflowState) -> MetricResult:
    """The calibrated judge as an (informational) metric for a new run."""
    v = judge(state)
    return MetricResult(name="expert_judge", score=1.0 if v.label == "pass" else 0.0,
                        passed=v.label == "pass", reason=v.critique)


# ---------------------------------------------------------------- calibration

def scores(gold: list[str], pred: list[str], positive: str = "fail") -> dict:
    tp = sum(g == positive and p == positive for g, p in zip(gold, pred))
    fp = sum(g != positive and p == positive for g, p in zip(gold, pred))
    fn = sum(g == positive and p != positive for g, p in zip(gold, pred))
    tn = len(gold) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if not tp + fp + fn:  # no positives anywhere and the judge agreed: perfect
        precision = recall = f1 = 1.0 if gold else 0.0
    return {"n": len(gold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "accuracy": round((tp + tn) / len(gold), 3) if gold else 0.0}


def calibrate(split: Split = "dev", path: Optional[Path] = None) -> dict:
    """Run the judge on one split and compare with the expert labels."""
    if split == "train":
        raise ValueError("train rows are the judge's few-shot examples; calibrate on dev, then test")
    rows = load_dataset(path)
    train = [r for r in rows if r.split == "train"]
    target = [r for r in rows if r.split == split]
    if not target:
        raise ValueError(f"No labelled rows in the {split} split yet; label more runs first.")
    results = []
    for r in target:
        st = load_state(r.run_dir)
        v = judge(st, _brief_of(st, r.version), train)
        results.append({"id": r.id, "query": r.query, "label": r.label, "predicted": v.label,
                        "agree": r.label == v.label, "expert_critique": r.critique, "judge_critique": v.critique})
    report = {"split": split, "train_examples": len(train), "judge_model": config.judge_model_name(),
              **scores([x["label"] for x in results], [x["predicted"] for x in results]), "rows": results}
    out = Path(config.EVAL_DATA_DIR) / f"calibration-{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["saved_to"] = str(out)
    return report


def print_report(report: dict) -> None:
    print(f"Split {report['split']}: n={report['n']}  F1(fail)={report['f1']}  precision={report['precision']}  "
          f"recall={report['recall']}  accuracy={report['accuracy']}  (train examples: {report['train_examples']})")
    for x in report["rows"]:
        if not x["agree"]:
            print(f"- DISAGREE {x['id']}: expert={x['label']} judge={x['predicted']}\n"
                  f"    expert: {x['expert_critique']}\n    judge:  {x['judge_critique']}")
    print(f"Saved {report.get('saved_to', '')}")
