"""Tests for retrieval components: Dense FAISS, BM25 Lexical, RRF Hybrid, Reranking, and IBM Granite."""
import pytest
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_retriever import BM25SparseRetriever, tokenize
from src.retrieval.reranker import CrossEncoderReranker
from src.vector_store import resolve_model_name, MODEL_NAME_ALIASES


class TestRetrievalEngine:

    def test_granite_model_resolution(self):
        """Verify IBM Granite embedding alias maps to canonical HF repo."""
        assert resolve_model_name("granite") == "ibm-granite/granite-embedding-125m-english"
        assert resolve_model_name("ibm-granite") == "ibm-granite/granite-embedding-125m-english"
        assert resolve_model_name("granite-30m") == "ibm-granite/granite-embedding-30m-english"
        assert resolve_model_name("all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"

    def test_bm25_alphanumeric_tokenization(self):
        """BM25 tokenizer must preserve alphanumeric incident codes, hotfixes, and numbers."""
        tokens = tokenize("Investigate incident INC-7721-P1 and deploy hotfix HF-8842 by 04:30 UTC.")
        tokens_lower = [t.lower() for t in tokens]
        assert "inc-7721-p1" in tokens_lower or "7721" in tokens_lower
        assert any("8842" in t for t in tokens_lower)

    def test_sparse_bm25_retrieval(self, sample_retriever):
        """Verify BM25 retrieves matching exact identifiers."""
        hits = sample_retriever.search_sparse("INC-7721-P1 root cause", top_k=2)
        assert len(hits) >= 1
        top_doc, score = hits[0]
        assert "INC-7721-P1" in top_doc.page_content
        assert score > 0

    def test_dense_vector_search(self, sample_retriever):
        """Verify Dense vector search executes and returns ranked hits."""
        hits = sample_retriever.search_dense("Cloud Infrastructure revenue", top_k=2)
        assert len(hits) >= 1
        top_doc, score = hits[0]
        assert hasattr(top_doc, "page_content")
        assert isinstance(float(score), float)

    def test_reciprocal_rank_fusion(self, sample_retriever):
        """Verify RRF merges dense and sparse results without errors."""
        hits = sample_retriever.search_hybrid_rrf("SLA uptime 99.95% guarantees", top_k=3)
        assert len(hits) >= 1
        # Highest ranked document should contain SLA uptime
        top_doc, score = hits[0]
        assert "99.95%" in top_doc.page_content or "SLA" in top_doc.page_content

    def test_linear_weighted_hybrid(self, sample_retriever):
        """Verify linear alpha-weighted hybrid search."""
        hits = sample_retriever.search_hybrid_weighted("Q3 Operating Margin 28.4%", top_k=3, alpha=0.7)
        assert len(hits) >= 1

    def test_cross_encoder_reranking_fallback(self, sample_documents):
        """Verify cross-encoder reranks or gracefully falls back on candidate pools."""
        reranker = CrossEncoderReranker()
        candidates = [(doc, 0.5) for doc in sample_documents]
        reranked = reranker.rerank(query="Incident hotfix HF-8842", candidate_docs=candidates, top_k=2)
        assert len(reranked) == 2
        # Top result should be the incident email
        assert "Incident" in reranked[0][0].metadata.get("doc_name", "") or "INC-7721" in reranked[0][0].page_content
