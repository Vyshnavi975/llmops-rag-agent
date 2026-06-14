"""Document ingestion: load raw text/markdown files, split them into chunks
with LangChain's `RecursiveCharacterTextSplitter`, and build a vector index.

This module intentionally does the loading by hand (a couple of lines of
`pathlib` code) rather than pulling in a directory-loader abstraction — for
a folder of local `.md`/`.txt` files that would be an extra dependency for
no real benefit. The parts that matter for an LLMOps/RAG portfolio piece —
chunking strategy and the vector store abstraction — use real LangChain
components (`RecursiveCharacterTextSplitter`, `Document`, `Embeddings`,
`VectorStore`).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import embeddings_backend_name, get_embeddings

SUPPORTED_EXTENSIONS = {".md", ".txt"}
INDEX_STORE_FILENAME = "vectorstore.json"
INDEX_META_FILENAME = "meta.json"


@dataclass
class IngestStats:
    num_source_documents: int
    num_chunks: int
    embedding_backend: str
    elapsed_seconds: float
    index_dir: str


def load_documents(docs_dir: str) -> List[Document]:
    """Read every supported file in `docs_dir` into a LangChain `Document`,
    tagging each with its source filename in metadata."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents: List[Document] = []
    for path in sorted(docs_path.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": path.name}))

    if not documents:
        raise ValueError(
            f"No supported documents ({sorted(SUPPORTED_EXTENSIONS)}) found in {docs_dir}"
        )
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """Split documents into overlapping chunks using LangChain's
    `RecursiveCharacterTextSplitter`, which tries a cascade of separators
    (paragraph, then line, then sentence, then word) so chunks break on
    natural boundaries where possible instead of mid-sentence."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    # Tag each chunk with a stable id (source file + index within that file)
    # so trace/eval output can cite exactly which chunk was retrieved.
    counts: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
        chunk.metadata["chunk_id"] = f"{source}#{counts[source]}"
    return chunks


def build_index(docs_dir: str, index_dir: str, chunk_size: int = 800, chunk_overlap: int = 120) -> IngestStats:
    """Load documents from `docs_dir`, chunk them, embed them, and persist a
    vector index to `index_dir`."""
    start = time.time()
    documents = load_documents(docs_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embeddings = get_embeddings()
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    store.dump(str(index_path / INDEX_STORE_FILENAME))

    meta = {
        "embedding_backend": embeddings_backend_name(),
        "num_source_documents": len(documents),
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "docs_dir": str(Path(docs_dir).resolve()),
        "built_at": time.time(),
    }
    (index_path / INDEX_META_FILENAME).write_text(json.dumps(meta, indent=2))

    elapsed = time.time() - start
    return IngestStats(
        num_source_documents=len(documents),
        num_chunks=len(chunks),
        embedding_backend=embeddings_backend_name(),
        elapsed_seconds=elapsed,
        index_dir=str(index_path.resolve()),
    )
