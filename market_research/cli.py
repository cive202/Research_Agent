"""Command-line entry points: `python -m market_research.cli {run,eval}`."""
from __future__ import annotations

import argparse
import json

from . import config
from .rendering import brief_to_markdown
from .workflow import orchestrator


def cmd_run(args: argparse.Namespace) -> None:
    result = orchestrator(args.query, use_sample=args.sample, on_step=print,
                          evaluate_rag=not args.no_eval, force_revision=args.force_revision, ragas=args.ragas,
                          save_dir=None if args.no_save else config.RUNS_DIR)
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="market_research")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="generate a research brief")
    r.add_argument("query")
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
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
