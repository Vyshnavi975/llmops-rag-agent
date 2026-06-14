"""Evaluation harness: run the labeled eval set through the full agent
graph and score every prediction with the metrics in `ops/metrics.py`.

This is what makes prompt/model changes in this project accountable to
numbers instead of vibes: change `answer_v1` to `answer_v2`, or swap
`gpt-4o-mini` for a bigger model, re-run `python cli.py eval`, and get a
before/after pass rate and per-example diff instead of eyeballing a couple
of manual test questions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from agent.graph import AgentGraphBundle, build_agent_graph, run_agent
from ops.metrics import ExampleScore, score_example

DEFAULT_EVAL_SET_PATH = "ops/eval_set.json"
DEFAULT_REPORTS_DIR = "reports"


@dataclass
class EvalReport:
    prompt_label: str
    model_name: str
    mode: str
    num_examples: int
    pass_rate: float
    avg_f1: float
    avg_rouge_l: float
    avg_keyword_recall: float
    total_retries_triggered: int
    total_cost_usd: float
    total_tokens: int
    elapsed_seconds: float
    example_scores: List[ExampleScore]

    def to_dict(self) -> dict:
        return {
            "prompt_label": self.prompt_label,
            "model_name": self.model_name,
            "mode": self.mode,
            "num_examples": self.num_examples,
            "pass_rate": round(self.pass_rate, 3),
            "avg_f1": round(self.avg_f1, 3),
            "avg_rouge_l": round(self.avg_rouge_l, 3),
            "avg_keyword_recall": round(self.avg_keyword_recall, 3),
            "total_retries_triggered": self.total_retries_triggered,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "examples": [ex.to_dict() for ex in self.example_scores],
        }


def load_eval_set(path: str = DEFAULT_EVAL_SET_PATH) -> List[dict]:
    return json.loads(Path(path).read_text())


def run_eval(
    index_dir: str,
    eval_set_path: str = DEFAULT_EVAL_SET_PATH,
    prompt_name: str = "answer",
    prompt_version: str | int = "latest",
    model: Optional[str] = None,
    traces_dir: str = "traces",
    save_traces: bool = True,
    pass_threshold: float = 0.5,
    bundle: Optional[AgentGraphBundle] = None,
) -> EvalReport:
    start = time.time()
    examples = load_eval_set(eval_set_path)

    if bundle is None:
        bundle = build_agent_graph(
            index_dir=index_dir, prompt_name=prompt_name, prompt_version=prompt_version, model=model
        )

    scores: List[ExampleScore] = []
    total_retries = 0
    total_cost = 0.0
    total_tokens = 0

    for example in examples:
        result = run_agent(bundle, example["question"], traces_dir=traces_dir, save_trace=save_traces)
        score = score_example(
            example_id=example["id"],
            question=example["question"],
            prediction=result.answer,
            reference=example["reference_answer"],
            key_facts=example.get("key_facts", []),
            pass_threshold=pass_threshold,
        )
        scores.append(score)
        if result.retrieval_count > 1:
            total_retries += 1
        total_cost += result.cost_summary.get("total_cost_usd", 0.0)
        total_tokens += result.cost_summary.get("total_tokens", 0)

    n = len(scores) or 1
    report = EvalReport(
        prompt_label=bundle.prompt_label,
        model_name=bundle.model_name,
        mode="demo" if bundle.is_demo else "live",
        num_examples=len(scores),
        pass_rate=sum(1 for s in scores if s.passed) / n,
        avg_f1=sum(s.f1 for s in scores) / n,
        avg_rouge_l=sum(s.rouge_l for s in scores) / n,
        avg_keyword_recall=sum(s.keyword_recall for s in scores) / n,
        total_retries_triggered=total_retries,
        total_cost_usd=total_cost,
        total_tokens=total_tokens,
        elapsed_seconds=time.time() - start,
        example_scores=scores,
    )
    return report


def save_report(report: EvalReport, reports_dir: str = DEFAULT_REPORTS_DIR) -> str:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    filename = f"eval_{report.prompt_label}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    file_path = reports_path / filename
    file_path.write_text(json.dumps(report.to_dict(), indent=2))
    return str(file_path)


def format_report_table(report: EvalReport) -> str:
    lines = []
    lines.append(f"Prompt: {report.prompt_label}   Model: {report.model_name} ({report.mode} mode)")
    lines.append(
        f"Pass rate: {report.pass_rate:.1%}   Avg F1: {report.avg_f1:.3f}   "
        f"Avg ROUGE-L: {report.avg_rouge_l:.3f}   Avg keyword recall: {report.avg_keyword_recall:.1%}"
    )
    lines.append(
        f"Retries triggered: {report.total_retries_triggered}/{report.num_examples}   "
        f"Total tokens: {report.total_tokens}   Total cost: ${report.total_cost_usd:.6f}   "
        f"Elapsed: {report.elapsed_seconds:.2f}s"
    )
    lines.append("")
    header = f"{'id':<10} {'pass':<6} {'f1':>6} {'rouge_l':>8} {'kw_recall':>10}  question"
    lines.append(header)
    lines.append("-" * len(header) if len(header) > 60 else "-" * 100)
    for ex in report.example_scores:
        d = ex.to_dict()
        status = "PASS" if d["passed"] else "FAIL"
        lines.append(
            f"{d['id']:<10} {status:<6} {d['f1']:>6.3f} {d['rouge_l']:>8.3f} {d['keyword_recall']:>10.1%}  {d['question'][:60]}"
        )
        if d["keyword_missed"]:
            lines.append(f"{'':<10} {'':<6} missed key facts: {d['keyword_missed']}")
    return "\n".join(lines)
