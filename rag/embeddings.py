"""Embeddings for the RAG pipeline.

Two implementations are exposed behind one factory function, `get_embeddings`:

* `OpenAIEmbeddings` (from `langchain-openai`) when `OPENAI_API_KEY` is set in
  the environment — real, high-quality embeddings suitable for production use.
* `HashingEmbeddings`, a small offline, deterministic embedding implemented
  from scratch below, used automatically when no API key is available so the
  whole project (ingest -> ask -> eval -> trace) runs with zero network
  access and zero cost. It still implements LangChain's `Embeddings`
  interface, so every downstream component (vector store, retriever) treats
  it exactly like any other embedding model.

`HashingEmbeddings` is a deterministic bag-of-words / hashing-trick vectorizer:
each token is hashed into a fixed-size vector slot (with a sign derived from
the hash to reduce collision bias, the standard "feature hashing" trick), the
result is L2-normalized, and a handful of lightweight character n-gram
features are mixed in so that near-duplicate tokens (e.g. "pipeline" vs
"pipelines") still land close together in vector space. It is not a neural
embedding and will not match OpenAI's semantic quality, but it is fully
deterministic (same text -> same vector, always) and requires no model
download, which is exactly what a demo/offline mode needs.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List

from langchain_core.embeddings import Embeddings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _char_ngrams(token: str, n: int = 3) -> List[str]:
    if len(token) <= n:
        return [token]
    return [token[i : i + n] for i in range(len(token) - n + 1)]


def _hash_to_index(feature: str, dim: int) -> int:
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dim


def _hash_sign(feature: str) -> float:
    digest = hashlib.sha256((feature + "|sign").encode("utf-8")).digest()
    return 1.0 if digest[0] % 2 == 0 else -1.0


class HashingEmbeddings(Embeddings):
    """Deterministic, offline, dependency-free embedding via the hashing trick.

    Implements LangChain's `Embeddings` interface (`embed_documents`,
    `embed_query`) so it is a drop-in replacement for `OpenAIEmbeddings`
    anywhere in the pipeline (vector store construction, retrievers, etc).
    """

    def __init__(self, dim: int = 512, use_char_ngrams: bool = True) -> None:
        self.dim = dim
        self.use_char_ngrams = use_char_ngrams

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = _tokenize(text)
        for tok in tokens:
            idx = _hash_to_index(tok, self.dim)
            vec[idx] += _hash_sign(tok)
            if self.use_char_ngrams:
                for ng in _char_ngrams(tok):
                    ng_idx = _hash_to_index("ngram:" + ng, self.dim)
                    vec[ng_idx] += 0.4 * _hash_sign("ngram:" + ng)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


def embeddings_backend_name() -> str:
    """Which embeddings backend `get_embeddings()` will select, without
    constructing it (used for logging/tracing)."""
    return "openai:text-embedding-3-small" if os.environ.get("OPENAI_API_KEY") else "offline:hashing-bow"


def get_embeddings() -> Embeddings:
    """Return a real OpenAI embedding model if an API key is configured,
    otherwise fall back to the offline deterministic hashing embedding.
    """
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HashingEmbeddings(dim=512)
