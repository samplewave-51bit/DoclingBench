"""Hybrid retrieval engine combining Dense Vector Search (FAISS) and Sparse Keyword Search (BM25) with RRF and Cross-Encoder Reranking."""
import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document

from src.retrieval.sparse_retriever import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.config import RRF_K_CONSTANT, DEFAULT_HYBRID_ALPHA, DEFAULT_RERANKER_MODEL

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Orchestrates multi-modal retrieval: Dense, BM25, RRF Fusion, Weighted Hybrid, and Reranking."""

    def __init__(
        self,
        vector_store,  # LectureVectorStore or DenseRetriever
        documents: List[Document] = None,
        reranker_model: str = DEFAULT_RERANKER_MODEL
    ):
        self.vector_store = vector_store
        self.sparse_retriever = BM25Retriever(documents) if documents else None
        self.reranker = CrossEncoderReranker(model_name=reranker_model)
        self.all_documents = documents or []

    def update_documents(self, documents: List[Document]):
        """Update indexed documents for sparse search."""
        self.all_documents = documents
        if not self.sparse_retriever:
            self.sparse_retriever = BM25Retriever(documents)
        else:
            self.sparse_retriever.index_documents(documents)

    def search_dense(self, query: str, top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Dense semantic search using vector embeddings."""
        if not self.vector_store or self.vector_store.is_empty:
            return []
        return self.vector_store.search(query, top_k=top_k, filter_dict=filter_dict)

    def search_sparse(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Sparse lexical search using BM25."""
        if not self.sparse_retriever:
            return []
        return self.sparse_retriever.search(query, top_k=top_k)

    def search_hybrid_rrf(
        self,
        query: str,
        candidate_k: int = 20,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        rrf_k: int = RRF_K_CONSTANT
    ) -> List[Tuple[Document, float]]:
        """Reciprocal Rank Fusion (RRF) between Dense and BM25 retrievers."""
        dense_results = self.search_dense(query, top_k=candidate_k, filter_dict=filter_dict)
        sparse_results = self.search_sparse(query, top_k=candidate_k)

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # 1. Score dense rankings
        for rank, (doc, _) in enumerate(dense_results, start=1):
            doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # 2. Score sparse rankings
        for rank, (doc, _) in enumerate(sparse_results, start=1):
            doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        return [(doc_map[doc_id], rrf_scores[doc_id]) for doc_id in sorted_ids[:top_k]]

    def search_hybrid_weighted(
        self,
        query: str,
        alpha: float = DEFAULT_HYBRID_ALPHA,
        candidate_k: int = 20,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Linear weighted combination of normalized Dense and BM25 scores."""
        dense_results = self.search_dense(query, top_k=candidate_k, filter_dict=filter_dict)
        sparse_results = self.search_sparse(query, top_k=candidate_k)

        # Min-max normalization helper
        def min_max_norm(results: List[Tuple[Document, float]]) -> Dict[str, Tuple[Document, float]]:
            if not results:
                return {}
            scores = [s for _, s in results]
            min_s, max_s = min(scores), max(scores)
            norm_map = {}
            for doc, s in results:
                doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
                norm_score = (s - min_s) / (max_s - min_s) if max_s > min_s else 1.0
                norm_map[doc_id] = (doc, norm_score)
            return norm_map

        dense_norm = min_max_norm(dense_results)
        sparse_norm = min_max_norm(sparse_results)

        all_doc_ids = set(dense_norm.keys()).union(set(sparse_norm.keys()))
        combined_scores: Dict[str, Tuple[Document, float]] = {}

        for doc_id in all_doc_ids:
            doc = dense_norm[doc_id][0] if doc_id in dense_norm else sparse_norm[doc_id][0]
            d_score = dense_norm[doc_id][1] if doc_id in dense_norm else 0.0
            s_score = sparse_norm[doc_id][1] if doc_id in sparse_norm else 0.0
            final_score = alpha * d_score + (1.0 - alpha) * s_score
            combined_scores[doc_id] = (doc, final_score)

        sorted_results = sorted(combined_scores.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def search_hybrid_reranked(
        self,
        query: str,
        candidate_pool_k: int = 20,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """First retrieve candidate pool via RRF, then re-rank with Cross-Encoder."""
        candidates = self.search_hybrid_rrf(query, candidate_k=candidate_pool_k, top_k=candidate_pool_k, filter_dict=filter_dict)
        if not candidates:
            return []
        
        candidate_docs = [doc for doc, _ in candidates]
        return self.reranker.rerank(query, candidate_docs, top_k=top_k)

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_rrf_reranked",
        top_k: int = 5,
        candidate_pool_k: int = 20,
        alpha: float = DEFAULT_HYBRID_ALPHA,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Unified retrieval entrypoint supporting all benchmark retrieval modes."""
        if mode == "dense_only":
            return self.search_dense(query, top_k=top_k, filter_dict=filter_dict)
        elif mode == "sparse_only":
            return self.search_sparse(query, top_k=top_k)
        elif mode == "hybrid_rrf":
            return self.search_hybrid_rrf(query, candidate_k=candidate_pool_k, top_k=top_k, filter_dict=filter_dict)
        elif mode == "hybrid_weighted":
            return self.search_hybrid_weighted(query, alpha=alpha, candidate_k=candidate_pool_k, top_k=top_k, filter_dict=filter_dict)
        elif mode == "hybrid_rrf_reranked":
            return self.search_hybrid_reranked(query, candidate_pool_k=candidate_pool_k, top_k=top_k, filter_dict=filter_dict)
        else:
            logger.warning(f"Unknown retrieval mode '{mode}'. Defaulting to hybrid_rrf_reranked.")
            return self.search_hybrid_reranked(query, candidate_pool_k=candidate_pool_k, top_k=top_k, filter_dict=filter_dict)
