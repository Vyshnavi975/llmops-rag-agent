"""The agentic RAG graph, built with LangGraph's `StateGraph`.

    START
      |
      v
+-----------------+
| decide_retrieval|  heuristic: does this question need the knowledge base?
+-----------------+
      |  needs_retrieval?
      |-------------------------------+
      | yes                           | no
      v                                v
 +----------+                   +-------------+
 |  retrieve| <--------retry---- |  generate   |
 +----------+                    +-------------+
      |                                 ^
      v                                 |
 +----------+                           |
 | generate | -------------------------->
 +----------+
      |
      v
 +------------+
 | self_check |
 +------------+
   |   passed OR retries exhausted --> END
   |   failed AND retries left    --> retrieve (broadened query, larger k)
   +-----------------------------------------------------------------+

`retrieve` -> `generate` -> `self_check` -> (loop back to `retrieve`, or END)
is a real conditional loop compiled by LangGraph, not a fixed pipeline: the
number of times `retrieve`/`generate` execute in a given run depends on
what `self_check` decides at runtime, bounded by `max_retries`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.demo_answers import build_demo_answer
from agent.heuristics import extract_keywords, looks_like_greeting_or_meta
from agent.llm import get_llm, set_fake_response
from agent.state import AgentState
from ops.cost import CostTracker
from ops.prompt_registry import PromptRegistry
from ops.tracing import RunTracer
from rag.embeddings import embeddings_backend_name
from rag.store import load_vectorstore

DEFAULT_K = 3
DEFAULT_MAX_RETRIES = 1
RETRY_K_INCREMENT = 3
GROUNDING_PASS_THRESHOLD = 0.5


@dataclass
class AgentGraphBundle:
    graph: CompiledStateGraph
    prompt_label: str
    model_name: str
    is_demo: bool
    embedding_backend: str
    default_k: int
    max_retries: int


def _to_documents(retrieved_docs: List[dict]) -> List[Document]:
    return [
        Document(page_content=d["content"], metadata={"source": d.get("source"), "chunk_id": d.get("chunk_id")})
        for d in retrieved_docs
    ]


def build_agent_graph(
    index_dir: str,
    prompt_name: str = "answer",
    prompt_version: str | int = "latest",
    prompts_dir: str = "prompts",
    model: Optional[str] = None,
    default_k: int = DEFAULT_K,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> AgentGraphBundle:
    """Compile the agentic RAG graph, bound to a specific vector index,
    prompt version, and model. Building a fresh bundle per configuration
    (rather than a single global graph) is what lets the eval harness run
    the exact same agent against `answer_v1` vs `answer_v2`, or one model
    vs another, and compare the results.
    """
    store = load_vectorstore(index_dir)
    registry = PromptRegistry(prompts_dir)
    resolved_version = registry.resolve_version(prompt_name, prompt_version)
    prompt_template = registry.get_template(prompt_name, resolved_version)
    prompt_label = registry.get_version_label(prompt_name, resolved_version)
    # Behavioral hook tied to the prompt version: v2+ asks for concise,
    # multi-fact, cited answers; v1 does not. In demo mode this drives the
    # extractive heuristic directly (see agent/demo_answers.py); with a real
    # LLM the instruction is simply part of the prompt text itself.
    cite_sources = resolved_version >= 2

    llm, model_name, is_demo = get_llm(model)

    def decide_retrieval_node(state: AgentState) -> Dict[str, Any]:
        t0 = time.time()
        question = state["question"]
        is_meta = looks_like_greeting_or_meta(question)
        needs = not is_meta
        reason = (
            "greeting/meta question detected -> skip retrieval"
            if is_meta
            else "informational question -> retrieval required"
        )
        latency_ms = (time.time() - t0) * 1000
        tracer: Optional[RunTracer] = state.get("tracer")
        if tracer:
            tracer.log_node(
                "decide_retrieval",
                input_summary=question,
                output_summary=f"needs_retrieval={needs} ({reason})",
                latency_ms=latency_ms,
            )
        return {
            "needs_retrieval": needs,
            "route_reason": reason,
            "query": question,
            "k": state.get("k") or default_k,
            "retrieval_count": 0,
            "max_retries": state.get("max_retries") if state.get("max_retries") is not None else max_retries,
        }

    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        t0 = time.time()
        query = state.get("query") or state["question"]
        k = state.get("k") or default_k
        results = store.similarity_search_with_score(query, k=k)
        retrieved = [
            {
                "source": doc.metadata.get("source"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content": doc.page_content,
                "score": float(score),
            }
            for doc, score in results
        ]
        context = "\n\n".join(f"[{r['source']}] {r['content']}" for r in retrieved)
        retrieval_count = state.get("retrieval_count", 0) + 1
        latency_ms = (time.time() - t0) * 1000
        tracer = state.get("tracer")
        if tracer:
            tracer.log_node(
                "retrieve",
                input_summary=f"query='{query}', k={k}",
                output_summary=f"{len(retrieved)} chunk(s) from {sorted({r['source'] for r in retrieved})}",
                latency_ms=latency_ms,
                extra={"retrieval_pass": retrieval_count, "k": k},
            )
        return {"retrieved_docs": retrieved, "context": context, "retrieval_count": retrieval_count}

    def generate_node(state: AgentState) -> Dict[str, Any]:
        t0 = time.time()
        question = state["question"]
        context = state.get("context", "")
        docs = state.get("retrieved_docs", [])
        prompt_text = prompt_template.format(
            context=context or "(no context retrieved — answer only if this is a general/meta question)",
            question=question,
        )

        if is_demo:
            demo_text = build_demo_answer(question, _to_documents(docs), cite_sources=cite_sources)
            set_fake_response(llm, demo_text)

        ai_message = llm.invoke(prompt_text)
        answer = ai_message.content

        cost_tracker: Optional[CostTracker] = state.get("cost_tracker")
        if cost_tracker is not None:
            cost_tracker.record(node="generate", prompt_text=prompt_text, completion_text=answer, model=model_name)

        latency_ms = (time.time() - t0) * 1000
        tracer = state.get("tracer")
        if tracer:
            tracer.log_node(
                "generate",
                input_summary=f"prompt={prompt_label}, context_chars={len(context)}",
                output_summary=answer,
                latency_ms=latency_ms,
                extra={"model": model_name},
            )
        return {"answer": answer}

    def self_check_node(state: AgentState) -> Dict[str, Any]:
        t0 = time.time()
        question = state["question"]
        context = state.get("context", "")
        answer = state.get("answer", "")
        needs_retrieval = state.get("needs_retrieval", True)
        retrieval_count = state.get("retrieval_count", 0)
        state_max_retries = state.get("max_retries", max_retries)

        if not needs_retrieval:
            passed, grounding, missing = True, 1.0, []
            reason = "no retrieval was needed for this question"
        else:
            keywords = extract_keywords(question)
            context_lower = context.lower()
            missing = [kw for kw in keywords if kw not in context_lower]
            grounding = 1.0 if not keywords else (len(keywords) - len(missing)) / len(keywords)
            says_dont_know = "i don't know" in answer.lower()
            passed = grounding >= GROUNDING_PASS_THRESHOLD and not says_dont_know
            reason = f"grounding_score={grounding:.2f} over {len(keywords)} question keyword(s)"
            if says_dont_know:
                reason += "; answer explicitly disclaimed knowledge"

        can_retry = (not passed) and needs_retrieval and retrieval_count < (1 + state_max_retries)

        latency_ms = (time.time() - t0) * 1000
        tracer = state.get("tracer")
        if tracer:
            tracer.log_node(
                "self_check",
                input_summary=f"answer='{answer[:80]}'",
                output_summary=f"passed={passed} ({reason}); retry={can_retry}",
                latency_ms=latency_ms,
                extra={"grounding_score": round(grounding, 3), "missing_keywords": missing},
            )

        update: Dict[str, Any] = {"self_check_passed": passed, "grounding_score": grounding, "missing_keywords": missing}
        if can_retry:
            update["query"] = question + " " + " ".join(missing)
            update["k"] = (state.get("k") or default_k) + RETRY_K_INCREMENT
        return update

    def route_after_decision(state: AgentState) -> str:
        return "retrieve" if state.get("needs_retrieval") else "generate"

    def route_after_self_check(state: AgentState) -> str:
        passed = state.get("self_check_passed", True)
        needs_retrieval = state.get("needs_retrieval", True)
        retrieval_count = state.get("retrieval_count", 0)
        state_max_retries = state.get("max_retries", max_retries)
        if not passed and needs_retrieval and retrieval_count < (1 + state_max_retries):
            return "retry"
        return "finish"

    builder = StateGraph(AgentState)
    builder.add_node("decide_retrieval", decide_retrieval_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("self_check", self_check_node)

    builder.add_edge(START, "decide_retrieval")
    builder.add_conditional_edges(
        "decide_retrieval", route_after_decision, {"retrieve": "retrieve", "generate": "generate"}
    )
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "self_check")
    builder.add_conditional_edges("self_check", route_after_self_check, {"retry": "retrieve", "finish": END})

    compiled = builder.compile()

    return AgentGraphBundle(
        graph=compiled,
        prompt_label=prompt_label,
        model_name=model_name,
        is_demo=is_demo,
        embedding_backend=embeddings_backend_name(),
        default_k=default_k,
        max_retries=max_retries,
    )


@dataclass
class AgentRunResult:
    answer: str
    trace_path: Optional[str]
    trace: dict
    retrieval_count: int
    self_check_passed: bool
    needs_retrieval: bool
    sources: List[str]
    cost_summary: dict


def run_agent(
    bundle: AgentGraphBundle,
    question: str,
    traces_dir: str = "traces",
    save_trace: bool = True,
) -> AgentRunResult:
    """Invoke a compiled agent graph for one question, with a fresh
    `RunTracer` and `CostTracker` for the run, and persist the resulting
    structured trace to disk."""
    tracer = RunTracer(
        meta={
            "question": question,
            "prompt": bundle.prompt_label,
            "model": bundle.model_name,
            "mode": "demo" if bundle.is_demo else "live",
            "embedding_backend": bundle.embedding_backend,
        }
    )
    cost_tracker = CostTracker(model=bundle.model_name)

    initial_state: AgentState = {
        "question": question,
        "k": bundle.default_k,
        "max_retries": bundle.max_retries,
        "retrieval_count": 0,
        "tracer": tracer,
        "cost_tracker": cost_tracker,
    }

    final_state = bundle.graph.invoke(initial_state, config={"recursion_limit": 25})

    cost_summary = cost_tracker.summary()
    tracer.finalize(final_output=final_state.get("answer", ""), cost_summary=cost_summary)
    trace_path = tracer.save(traces_dir) if save_trace else None

    sources = sorted({d["source"] for d in final_state.get("retrieved_docs", []) if d.get("source")})

    return AgentRunResult(
        answer=final_state.get("answer", ""),
        trace_path=trace_path,
        trace=tracer.to_dict(),
        retrieval_count=final_state.get("retrieval_count", 0),
        self_check_passed=final_state.get("self_check_passed", True),
        needs_retrieval=final_state.get("needs_retrieval", True),
        sources=sources,
        cost_summary=cost_summary,
    )
