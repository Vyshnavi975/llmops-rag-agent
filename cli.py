#!/usr/bin/env python3
"""Command-line interface for the LLMOps RAG Agent.

Subcommands:
    ingest --docs DIR [--index-dir data/index]
    ask --question "..." [--prompt-version latest] [--index-dir data/index]
    eval [--prompt-version latest] [--index-dir data/index]
    trace [--run-id ID]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from agent.graph import build_agent_graph, run_agent
from ops.eval_harness import format_report_table, run_eval, save_report
from ops.tracing import format_trace_pretty, get_latest_trace_path, load_trace
from rag.ingest import build_index

DEFAULT_INDEX_DIR = "data/index"
DEFAULT_DOCS_DIR = "sample_docs"


def cmd_ingest(args: argparse.Namespace) -> None:
    stats = build_index(
        docs_dir=args.docs,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Ingested {stats.num_source_documents} document(s) into {stats.num_chunks} chunk(s).")
    print(f"Embedding backend: {stats.embedding_backend}")
    print(f"Index written to:  {stats.index_dir}")
    print(f"Elapsed:            {stats.elapsed_seconds:.3f}s")


def cmd_ask(args: argparse.Namespace) -> None:
    bundle = build_agent_graph(
        index_dir=args.index_dir,
        prompt_name=args.prompt_name,
        prompt_version=args.prompt_version,
        model=args.model,
        default_k=args.k,
        max_retries=args.max_retries,
    )
    result = run_agent(bundle, args.question, traces_dir=args.traces_dir)

    print(f"Mode:            {'demo (offline)' if bundle.is_demo else 'live'}   Model: {bundle.model_name}")
    print(f"Prompt:          {bundle.prompt_label}")
    print(f"Needs retrieval: {result.needs_retrieval}")
    print(f"Retrieval passes: {result.retrieval_count}   Self-check passed: {result.self_check_passed}")
    if result.sources:
        print(f"Sources:         {', '.join(result.sources)}")
    print()
    print("Answer:")
    print(result.answer)
    print()
    cost = result.cost_summary
    print(
        f"Tokens: {cost['total_tokens']} (in={cost['total_input_tokens']}, out={cost['total_output_tokens']})   "
        f"Cost: ${cost['total_cost_usd']:.6f}"
    )
    print(f"Trace saved to:  {result.trace_path}")


def cmd_eval(args: argparse.Namespace) -> None:
    report = run_eval(
        index_dir=args.index_dir,
        eval_set_path=args.eval_set,
        prompt_name=args.prompt_name,
        prompt_version=args.prompt_version,
        model=args.model,
        traces_dir=args.traces_dir,
        save_traces=not args.no_traces,
    )
    print(format_report_table(report))
    report_path = save_report(report, reports_dir=args.reports_dir)
    print()
    print(f"Full JSON report saved to: {report_path}")


def cmd_trace(args: argparse.Namespace) -> None:
    if args.run_id:
        path = os.path.join(args.traces_dir, f"{args.run_id}.json")
    else:
        path = get_latest_trace_path(args.traces_dir)
    if not path or not os.path.exists(path):
        print("No trace found. Run `python cli.py ask ...` or `python cli.py eval` first.", file=sys.stderr)
        sys.exit(1)
    trace = load_trace(path)
    if args.json:
        print(json.dumps(trace, indent=2))
    else:
        print(format_trace_pretty(trace))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmops-rag-agent", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser("ingest", help="Chunk and embed documents into a vector index.")
    p_ingest.add_argument("--docs", default=DEFAULT_DOCS_DIR, help="Directory of .md/.txt documents to ingest.")
    p_ingest.add_argument("--index-dir", default=DEFAULT_INDEX_DIR, help="Where to write the vector index.")
    p_ingest.add_argument("--chunk-size", type=int, default=800)
    p_ingest.add_argument("--chunk-overlap", type=int, default=120)
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = subparsers.add_parser("ask", help="Run the agent graph on a single question.")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    p_ask.add_argument("--prompt-name", default="answer")
    p_ask.add_argument("--prompt-version", default="latest")
    p_ask.add_argument("--model", default=None, help="OpenAI chat model to use in live mode (default gpt-4o-mini).")
    p_ask.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve per retrieval pass.")
    p_ask.add_argument("--max-retries", type=int, default=1, help="Max self-check-triggered retrieval retries.")
    p_ask.add_argument("--traces-dir", default="traces")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = subparsers.add_parser("eval", help="Run the labeled eval set against the agent and print a report.")
    p_eval.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    p_eval.add_argument("--eval-set", default="ops/eval_set.json")
    p_eval.add_argument("--prompt-name", default="answer")
    p_eval.add_argument("--prompt-version", default="latest")
    p_eval.add_argument("--model", default=None)
    p_eval.add_argument("--traces-dir", default="traces")
    p_eval.add_argument("--reports-dir", default="reports")
    p_eval.add_argument("--no-traces", action="store_true", help="Skip writing a trace file per eval example.")
    p_eval.set_defaults(func=cmd_eval)

    p_trace = subparsers.add_parser("trace", help="Pretty-print the most recent (or a specific) run's trace.")
    p_trace.add_argument("--run-id", default=None, help="Specific run id; defaults to the most recent run.")
    p_trace.add_argument("--traces-dir", default="traces")
    p_trace.add_argument("--json", action="store_true", help="Print raw JSON instead of the formatted view.")
    p_trace.set_defaults(func=cmd_trace)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
