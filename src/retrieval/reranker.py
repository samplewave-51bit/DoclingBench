"""Cross-encoder reranking module for second-stage candidate re-scoring."""
import logging
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_RERANKER_CACHE: Dict[str, Any] = {}

class CrossEncoderReranker:
    """Reranks candidate documents using a pre-trained cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _get_model(self):
        if self.model_name not in _RERANKER_CACHE:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder '{self.model_name}' on {self.device}...")
                _RERANKER_CACHE[self.model_name] = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder '{self.model_name}': {e}. Using heuristic reranker.")
                _RERANKER_CACHE[self.model_name] = None
        return _RERANKER_CACHE[self.model_name]

    def rerank(self, query: str, candidates: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Score (query, chunk) pairs and return top_k reranked results."""
        if not candidates:
            return []

        model = self._get_model()
        if model is not None:
            try:
                pairs = [[query, doc.page_content] for doc in candidates]
                scores = model.predict(pairs)
                # Sort descending
                ranked_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                return [(doc, float(score)) for doc, score in ranked_pairs[:top_k]]
            except Exception as e:
                logger.error(f"Error during cross-encoder prediction: {e}")

        # Fallback heuristic: exact keyword overlap + length normalization
        query_words = set(query.lower().split())
        scored = []
        for doc in candidates:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(doc_words)) / max(len(query_words), 1)
            scored.append((doc, float(overlap)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
