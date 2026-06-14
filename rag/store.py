"""Loading a previously-built vector index and turning it into a retriever.

Vector store choice
--------------------
This project uses `langchain_core.vectorstores.InMemoryVectorStore` rather
than FAISS. Both were evaluated:

* `faiss-cpu` installs fine in this environment, but the LangChain FAISS
  integration currently only lives in `langchain-community`, which is being
  sunset upstream (deprecation warning on import) in favor of standalone
  integration packages. For a project meant to demonstrate current best
  practice, depending on a wrapper the maintainers are actively moving away
  from is the wrong trade-off when a first-class alternative exists.
* `InMemoryVectorStore` lives in `langchain-core` itself (no extra native
  dependency, no deprecation warning), implements the same `VectorStore`
  interface (`add_documents`, `similarity_search`, `similarity_search_with_score`,
  `as_retriever`, ...), and ships built-in JSON persistence (`dump`/`load`).
  For a corpus of this size (a few dozen chunks) its brute-force cosine
  search is also plenty fast — there is no accuracy or latency trade-off
  being made here, only a dependency and maintenance one.

Swapping in FAISS (or a hosted vector DB) later is a one-line change: only
this module and `rag/ingest.py` construct the vector store directly, and
both go through the same `Embeddings` interface, so nothing above them in
the agent graph would need to change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever

from rag.embeddings import get_embeddings
from rag.ingest import INDEX_META_FILENAME, INDEX_STORE_FILENAME


class IndexNotFoundError(RuntimeError):
    pass


def load_vectorstore(index_dir: str) -> InMemoryVectorStore:
    index_path = Path(index_dir)
    store_file = index_path / INDEX_STORE_FILENAME
    if not store_file.exists():
        raise IndexNotFoundError(
            f"No index found at '{index_dir}'. Run `python cli.py ingest --docs sample_docs` first."
        )
    embeddings = get_embeddings()
    return InMemoryVectorStore.load(str(store_file), embeddings)


def load_index_meta(index_dir: str) -> dict:
    meta_file = Path(index_dir) / INDEX_META_FILENAME
    if not meta_file.exists():
        return {}
    return json.loads(meta_file.read_text())


def get_retriever(index_dir: str, k: int = 4) -> VectorStoreRetriever:
    store = load_vectorstore(index_dir)
    return store.as_retriever(search_kwargs={"k": k})


def similarity_search_with_score(index_dir: str, query: str, k: int = 4) -> List[tuple[Document, float]]:
    store = load_vectorstore(index_dir)
    return store.similarity_search_with_score(query, k=k)
