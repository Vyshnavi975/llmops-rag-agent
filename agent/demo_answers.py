"""Deterministic, context-aware answer generation for demo mode.

In demo mode the "generation" LLM call is a `FakeListChatModel` (see
`agent/llm.py`), which simply replays whatever text it is handed. This
module computes that text: a small extractive-summarization heuristic that
picks the sentences from the retrieved context most relevant to the
question, so the graph produces a genuinely different, genuinely
context-dependent answer for every question and every retrieval — the
"fake" part is the LLM's reasoning, not the retrieval-grounding behavior.

This also gives prompt versioning something real to demonstrate: `answer_v1`
selects a single best sentence with no citation, `answer_v2` selects up to
three top sentences and appends a source citation. That is a genuine,
measurable difference the eval harness can score (see README for the
resulting numbers), standing in for the kind of behavioral change a real
prompt edit would cause with a real LLM.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document

from agent.heuristics import clean_markdown_for_display, extract_keywords, split_sentences

NO_CONTEXT_MESSAGE = (
    "I can answer questions about Nimbus Cloud's products, pricing, security "
    "and compliance, onboarding and support, API, and incident response "
    "processes. Please ask a specific question and I'll look it up."
)

UNGROUNDED_MESSAGE = (
    "I don't know based on the available documentation — none of the "
    "retrieved context appears to address that question."
)


def _score_sentences(question: str, docs: List[Document]) -> List[Tuple[float, str, str]]:
    """Return (score, source, sentence) tuples for every sentence in every
    retrieved doc, scored by how many question keywords it contains."""
    keywords = extract_keywords(question)
    scored: List[Tuple[float, str, str]] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        cleaned = clean_markdown_for_display(doc.page_content)
        for sentence in split_sentences(cleaned):
            sentence_lower = sentence.lower()
            score = sum(1 for kw in keywords if kw in sentence_lower)
            if score > 0:
                scored.append((score, source, sentence))
    return scored


def build_demo_answer(question: str, docs: List[Document], cite_sources: bool) -> str:
    """Build a deterministic, context-grounded answer string.

    - No retrieved docs at all -> a fixed "here's what I can help with" message
      (used on the no-retrieval / greeting branch).
    - Retrieved docs but no sentence overlaps the question's keywords -> an
      explicit "I don't know" message. This is what lets the self-check node
      detect an ungrounded answer and trigger a retry with broadened
      retrieval — a real, observable failure mode, not a scripted one.
    - Otherwise -> the top-scoring sentence(s), optionally with a source
      citation appended (prompt-version-dependent, see module docstring).
    """
    if not docs:
        return NO_CONTEXT_MESSAGE

    scored = _score_sentences(question, docs)
    if not scored:
        return UNGROUNDED_MESSAGE

    scored.sort(key=lambda item: item[0], reverse=True)
    top_n = 3 if cite_sources else 1
    top = scored[:top_n]

    sentences = [s for _, _, s in top]
    answer = " ".join(sentences)

    if cite_sources:
        sources = []
        for _, source, _ in top:
            if source not in sources:
                sources.append(source)
        answer += " [" + ", ".join(sources) + "]"

    return answer
