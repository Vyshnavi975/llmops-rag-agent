"""LangGraph state schema for the agentic RAG graph.

Kept as a plain `TypedDict` (LangGraph's standard state schema type): each
node function receives the current state and returns a partial dict of
fields to update, which LangGraph merges into the running state before the
next node runs. `total=False` because most nodes only touch a subset of
fields.
"""

from __future__ import annotations

from typing import Any, List, TypedDict


class RetrievedDoc(TypedDict, total=False):
    source: str
    chunk_id: str
    content: str
    score: float


class AgentState(TypedDict, total=False):
    # Input
    question: str

    # Routing
    needs_retrieval: bool
    route_reason: str

    # Retrieval
    query: str
    k: int
    retrieval_count: int
    max_retries: int
    retrieved_docs: List[RetrievedDoc]
    context: str

    # Generation
    answer: str

    # Self-check / reflection
    self_check_passed: bool
    grounding_score: float
    missing_keywords: List[str]

    # Run-scoped tooling (not business data, but simplest place for a
    # LangGraph node to reach a shared tracer/cost-tracker instance)
    tracer: Any
    cost_tracker: Any
