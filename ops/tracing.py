"""Structured run tracing.

Every agent run produces one JSON trace file describing exactly what the
LangGraph state machine did: which nodes it entered (including how many
times — self-check retries mean a node can run more than once in a single
run), a short summary of each node's input/output, per-node latency, and
per-node token/cost figures pulled from a `CostTracker`. This is the kind
of artifact that, in a real production system, would be the first thing an
engineer opens when a user reports "the bot gave a weird answer" — it turns
"the agent did something" into an inspectable, greppable record.

Traces are written to `traces/<run_id>.json`, one file per run, plus a
small `traces/_latest.txt` pointer file so the CLI's `trace` command can
find the most recent run without scanning directory mtimes.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_TRACES_DIR = "traces"
LATEST_POINTER_FILENAME = "_latest.txt"


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _summarize(text: str, max_len: int = 220) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


@dataclass
class NodeEvent:
    node: str
    seq: int
    input_summary: str
    output_summary: str
    latency_ms: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "seq": self.seq,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "latency_ms": round(self.latency_ms, 2),
            **self.extra,
        }


class RunTracer:
    """Collects `NodeEvent`s for a single agent run and persists them as
    structured JSON. One `RunTracer` is created per `ask`/eval-example
    invocation of the agent graph."""

    def __init__(self, meta: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or _new_run_id()
        self.meta = meta or {}
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.events: List[NodeEvent] = []
        self._seq = 0
        self.final_output: Optional[str] = None
        self.cost_summary: Optional[dict] = None

    def log_node(
        self,
        node: str,
        input_summary: str,
        output_summary: str,
        latency_ms: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> NodeEvent:
        self._seq += 1
        event = NodeEvent(
            node=node,
            seq=self._seq,
            input_summary=_summarize(input_summary),
            output_summary=_summarize(output_summary),
            latency_ms=latency_ms,
            extra=extra or {},
        )
        self.events.append(event)
        return event

    def finalize(self, final_output: str, cost_summary: Optional[dict] = None) -> None:
        self.finished_at = time.time()
        self.final_output = final_output
        self.cost_summary = cost_summary

    def to_dict(self) -> dict:
        total_latency_ms = sum(e.latency_ms for e in self.events)
        wall_clock_ms = (
            (self.finished_at - self.started_at) * 1000 if self.finished_at else None
        )
        return {
            "run_id": self.run_id,
            "meta": self.meta,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_clock_ms": round(wall_clock_ms, 2) if wall_clock_ms is not None else None,
            "total_node_latency_ms": round(total_latency_ms, 2),
            "num_node_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
            "final_output": self.final_output,
            "cost_summary": self.cost_summary,
        }

    def save(self, traces_dir: str = DEFAULT_TRACES_DIR) -> str:
        traces_path = Path(traces_dir)
        traces_path.mkdir(parents=True, exist_ok=True)
        file_path = traces_path / f"{self.run_id}.json"
        file_path.write_text(json.dumps(self.to_dict(), indent=2))
        (traces_path / LATEST_POINTER_FILENAME).write_text(self.run_id)
        return str(file_path)


def get_latest_trace_path(traces_dir: str = DEFAULT_TRACES_DIR) -> Optional[str]:
    traces_path = Path(traces_dir)
    pointer = traces_path / LATEST_POINTER_FILENAME
    if pointer.exists():
        run_id = pointer.read_text().strip()
        candidate = traces_path / f"{run_id}.json"
        if candidate.exists():
            return str(candidate)
    # Fall back to scanning for the most recently modified trace file, in
    # case the pointer file is missing or stale.
    trace_files = sorted(traces_path.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return str(trace_files[-1]) if trace_files else None


def load_trace(path: str) -> dict:
    return json.loads(Path(path).read_text())


def format_trace_pretty(trace: dict) -> str:
    lines: List[str] = []
    meta = trace.get("meta", {})
    lines.append(f"Run ID:        {trace['run_id']}")
    if meta:
        for key, value in meta.items():
            lines.append(f"  {key}: {value}")
    lines.append(f"Wall clock:    {trace.get('wall_clock_ms')} ms")
    lines.append(f"Node events:   {trace.get('num_node_events')}")
    lines.append("")
    lines.append(f"{'#':<3} {'node':<16} {'latency_ms':>10}  input -> output")
    lines.append("-" * 100)
    for event in trace.get("events", []):
        lines.append(
            f"{event['seq']:<3} {event['node']:<16} {event['latency_ms']:>10}  "
            f"{event['input_summary']}"
        )
        lines.append(f"    -> {event['output_summary']}")
        extras = {
            k: v
            for k, v in event.items()
            if k not in {"node", "seq", "input_summary", "output_summary", "latency_ms"}
        }
        if extras:
            lines.append(f"    [{extras}]")
    lines.append("")
    lines.append(f"Final answer: {_summarize(trace.get('final_output') or '', max_len=400)}")
    cost = trace.get("cost_summary") or {}
    if cost:
        lines.append("")
        lines.append(
            f"Cost:          {cost.get('num_llm_calls')} LLM call(s), "
            f"{cost.get('total_tokens')} tokens, ${cost.get('total_cost_usd')}"
        )
    return "\n".join(lines)
