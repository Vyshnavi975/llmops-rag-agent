"""Unit tests for the RAG ingestion + vector store pipeline. Runs fully
offline: no `OPENAI_API_KEY` is required or used (the fixture explicitly
unsets it so these tests are reproducible regardless of the host
environment), which routes `get_embeddings()` to the deterministic
`HashingEmbeddings` implementation.
"""

from pathlib import Path

import pytest

from rag.embeddings import HashingEmbeddings
from rag.ingest import build_index, chunk_documents, load_documents
from rag.store import IndexNotFoundError, load_index_meta, load_vectorstore

SAMPLE_DOCS_DIR = str(Path(__file__).resolve().parent.parent / "sample_docs")


@pytest.fixture(autouse=True)
def no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(scope="module")
def built_index(tmp_path_factory):
    index_dir = str(tmp_path_factory.mktemp("index"))
    build_index(docs_dir=SAMPLE_DOCS_DIR, index_dir=index_dir)
    return index_dir


class TestHashingEmbeddings:
    def test_same_text_gives_same_vector(self):
        emb = HashingEmbeddings(dim=64)
        assert emb.embed_query("hello world") == emb.embed_query("hello world")

    def test_different_text_gives_different_vector(self):
        emb = HashingEmbeddings(dim=64)
        assert emb.embed_query("hello world") != emb.embed_query("something else entirely")

    def test_vectors_are_unit_normalized(self):
        emb = HashingEmbeddings(dim=64)
        vec = emb.embed_query("nimbus cloud pricing plans")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-9

    def test_empty_text_returns_zero_vector_without_error(self):
        emb = HashingEmbeddings(dim=32)
        vec = emb.embed_query("")
        assert vec == [0.0] * 32

    def test_embed_documents_matches_embed_query_per_item(self):
        emb = HashingEmbeddings(dim=32)
        docs = ["first document", "second document"]
        batch = emb.embed_documents(docs)
        assert batch[0] == emb.embed_query("first document")
        assert batch[1] == emb.embed_query("second document")


class TestLoadAndChunk:
    def test_load_documents_finds_all_sample_docs(self):
        docs = load_documents(SAMPLE_DOCS_DIR)
        sources = {d.metadata["source"] for d in docs}
        assert "pricing_and_billing.md" in sources
        assert "security_and_compliance.md" in sources
        assert len(docs) == 6

    def test_load_documents_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path/for/sure")

    def test_chunking_produces_more_chunks_than_documents(self):
        docs = load_documents(SAMPLE_DOCS_DIR)
        chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=120)
        assert len(chunks) >= len(docs)

    def test_chunks_are_tagged_with_stable_chunk_ids(self):
        docs = load_documents(SAMPLE_DOCS_DIR)
        chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=120)
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert chunk.metadata["chunk_id"].startswith(chunk.metadata["source"])

    def test_smaller_chunk_size_produces_more_chunks(self):
        docs = load_documents(SAMPLE_DOCS_DIR)
        few = chunk_documents(docs, chunk_size=2000, chunk_overlap=100)
        many = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(many) > len(few)


class TestBuildAndLoadIndex:
    def test_build_index_reports_sane_stats(self, built_index):
        meta = load_index_meta(built_index)
        assert meta["num_source_documents"] == 6
        assert meta["num_chunks"] > 6
        assert meta["embedding_backend"] == "offline:hashing-bow"

    def test_loading_missing_index_raises(self, tmp_path):
        with pytest.raises(IndexNotFoundError):
            load_vectorstore(str(tmp_path / "no_index_here"))

    def test_similarity_search_finds_relevant_document(self, built_index):
        store = load_vectorstore(built_index)
        results = store.similarity_search_with_score("SOC 2 Type II audit certification compliance", k=3)
        assert len(results) == 3
        top_sources = {doc.metadata["source"] for doc, _score in results}
        assert "security_and_compliance.md" in top_sources

    def test_similarity_search_pricing_query_hits_pricing_doc(self, built_index):
        store = load_vectorstore(built_index)
        results = store.similarity_search_with_score("Team plan price per user per month billing", k=3)
        top_sources = {doc.metadata["source"] for doc, _score in results}
        assert "pricing_and_billing.md" in top_sources
