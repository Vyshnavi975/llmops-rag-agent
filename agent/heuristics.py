"""Small, shared, dependency-free text heuristics used by more than one
graph node: keyword extraction (used by both the demo answer generator and
the self-check grounding node) and a stopword list.

These are intentionally simple — the point of this project is the LLMOps
scaffolding and the LangGraph control flow around the LLM, not a novel NLP
technique. In a real deployment with a real LLM available, the "decide
retrieval" and "self-check" nodes could instead issue a cheap LLM call; the
heuristic version keeps the demo fully offline and, as a side benefit,
keeps routing/verification cost at $0 even in production, reserving the
paid LLM call for the one step that actually needs it (generation).
"""

from __future__ import annotations

import re
from typing import List, Set

_TOKEN_RE = re.compile(r"[a-z0-9%.$]+")

STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "i", "you", "he", "she",
    "it", "we", "they", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "will", "would", "should", "could", "can",
    "may", "might", "must", "shall", "to", "of", "in", "on", "for", "with",
    "at", "by", "from", "up", "about", "into", "over", "after", "and",
    "or", "but", "if", "than", "so", "as", "not", "no", "how", "when",
    "where", "why", "does", "your", "my", "our", "their", "me", "us",
    "does", "get", "gets", "getting", "much", "many",
}

_GREETING_FIRST_WORDS = {"hi", "hello", "hey", "thanks", "thank", "yo", "sup"}
_GREETING_FIRST_TWO_WORDS = {"good morning", "good afternoon", "good evening"}
_META_PHRASES = [
    "what can you do",
    "what can you help",
    "who are you",
    "can you help me",
    "what is this tool",
    "what is this assistant",
]


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def extract_keywords(text: str, min_len: int = 4) -> List[str]:
    """Content-bearing tokens: not a stopword, at least `min_len` chars
    (short numbers/percentages like "99.9%" are kept regardless of length
    since they're often the exact fact being asked about)."""
    keywords = []
    for tok in tokenize(text):
        is_numeric_ish = any(ch.isdigit() for ch in tok)
        if tok in STOPWORDS:
            continue
        if is_numeric_ish or len(tok) >= min_len:
            keywords.append(tok)
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def looks_like_greeting_or_meta(question: str) -> bool:
    """True for small-talk/meta questions ("hi", "thanks!", "what can you
    help with?") that don't need a knowledge-base lookup at all — this is
    the heuristic behind the `decide_retrieval` graph node's routing
    decision. Matches on the leading word(s) rather than the whole string
    so a full sentence like "Hi there, what can you help me with?" is still
    recognized, not just the bare word "hi"."""
    normalized = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    words = normalized.split()
    if not words:
        return False
    if words[0] in _GREETING_FIRST_WORDS:
        return True
    if " ".join(words[:2]) in _GREETING_FIRST_TWO_WORDS:
        return True
    joined = " ".join(words)
    return any(phrase in joined for phrase in _META_PHRASES)


def split_sentences(text: str) -> List[str]:
    """A plain regex sentence splitter — good enough for the short,
    well-formatted knowledge-base documents used here."""
    raw = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [s.strip() for s in raw if s.strip()]


_MD_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}[^\n]*$", re.MULTILINE)
_MD_BOLD_ITALIC_RE = re.compile(r"\*{1,3}")
_MD_BULLET_RE = re.compile(r"^\s*[-*]\s+", re.MULTILINE)


def clean_markdown_for_display(text: str) -> str:
    """Strip markdown heading *lines* (they read as sentence fragments, not
    sentences, so keeping them causes an extractive summarizer to glue a
    heading onto the following sentence as one run-on), bullet dashes, and
    bold/italic asterisks — so extractive answers built from raw markdown
    chunks read as plain prose instead of leaking source-document
    formatting."""
    text = _MD_HEADING_LINE_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    text = _MD_BOLD_ITALIC_RE.sub("", text)
    return text
