"""Command-line entry points: `python -m market_research {run,eval,label,calibrate,judge}`."""
from __future__ import annotations

import argparse
import json

from . import config
from .rendering import brief_to_markdown
from .workflow import orchestrator


def cmd_run(args: argparse.Namespace) -> None:
    from .guideline import load_guideline

    if not args.query and not args.guideline:
        raise SystemExit("Give a query or --guideline FILE.")
    guideline = load_guideline(args.guideline) if args.guideline else None
    result = orchestrator(args.query or "", use_sample=args.sample, on_step=print,
                          evaluate_rag=not args.no_eval, force_revision=args.force_revision, ragas=args.ragas,
                          save_dir=None if args.no_save else config.RUNS_DIR, guideline=guideline,
                          iterations=args.iterations)
    md = brief_to_markdown(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Saved to {args.out}")
    else:
        print("\n" + md)


def cmd_eval(args: argparse.Namespace) -> None:
    from .evaluation.batch import print_table, run_batch

    rows = run_batch(args.queries, args.sample)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print_table(rows)
    print(f"\nSaved {args.out}")


def cmd_label(args: argparse.Namespace) -> None:
    from .evaluation.calibration import add_label

    row = add_label(args.run_dir, args.label, args.critique, args.split, args.version)
    print(f"Labelled {row.id}: {row.label} ({row.split} split)")


def cmd_calibrate(args: argparse.Namespace) -> None:
    from .evaluation.calibration import calibrate, print_report

    print_report(calibrate(args.split))


def cmd_judge(args: argparse.Namespace) -> None:
    from .evaluation.calibration import judge, load_state

    v = judge(load_state(args.run_dir))
    print(f"{v.label.upper()}: {v.critique}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="market_research")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="generate a research brief")
    r.add_argument("query", nargs="?", help="company or market (optional with --guideline)")
    r.add_argument("--guideline", help="guideline markdown file (see guidelines/TEMPLATE.md)")
    r.add_argument("--iterations", type=int, default=config.REVIEW_ITERATIONS,
                   help=f"max review/edit rounds (default {config.REVIEW_ITERATIONS})")
    r.add_argument("--sample", action="store_true", help="use fixed sample data instead of web search")
    r.add_argument("--no-eval", action="store_true", help="skip the RAG evaluation")
    r.add_argument("--force-revision", action="store_true", help="force the critic to request a revision (test)")
    r.add_argument("--ragas", action="store_true", help="also compute RAGAS metrics (extra LLM calls)")
    r.add_argument("--no-save", action="store_true", help=f"do not write run files to {config.RUNS_DIR}/")
    r.add_argument("--out", help="write the brief to this markdown file")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("eval", help="run the batch evaluation")
    e.add_argument("--queries", default="test_queries.json")
    e.add_argument("--sample", action="store_true")
    e.add_argument("--out", default="eval_results.json")
    e.set_defaults(func=cmd_eval)

    lb = sub.add_parser("label", help="add your pass/fail label for a run to the judge dataset")
    lb.add_argument("run_dir", help="run folder with state.json, e.g. runs/20250101-120000-nepal-food")
    lb.add_argument("--label", required=True, choices=["pass", "fail"])
    lb.add_argument("--critique", required=True, help="2-3 sentences: why; for a fail, the first real problem")
    lb.add_argument("--split", choices=["train", "dev", "test"], help="default: assigned from a hash of the id")
    lb.add_argument("--version", type=int, help="label draft_vN instead of the final brief")
    lb.set_defaults(func=cmd_label)

    c = sub.add_parser("calibrate", help="measure the judge against your labels (F1 on dev or test)")
    c.add_argument("--split", default="dev", choices=["dev", "test"])
    c.set_defaults(func=cmd_calibrate)

    j = sub.add_parser("judge", help="run the calibrated judge on one run")
    j.add_argument("run_dir")
    j.set_defaults(func=cmd_judge)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
