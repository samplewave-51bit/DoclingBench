"""Sparse lexical retrieval using BM25Okapi for keyword, jargon, and code matching."""
import re
import logging
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", 
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", 
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", 
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", 
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", 
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", 
    "you", "your", "yours", "yourself", "yourselves"
}

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words, keeping hyphenated codes intact."""
    tokens = re.findall(r'\b[a-zA-Z0-9_\-\.\$]{2,}\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25Retriever:
    """In-memory BM25 index over processed chunks."""

    def __init__(self, documents: List[Document] = None):
        self.documents: List[Document] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi = None
        if documents:
            self.index_documents(documents)

    def index_documents(self, documents: List[Document]):
        """Index a list of documents."""
        self.documents = documents
        self.corpus_tokens = [tokenize(doc.page_content) for doc in documents]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            logger.info(f"Indexed {len(documents)} chunks in BM25 index.")
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve top_k documents with raw BM25 score."""
        if not self.bm25 or not self.documents:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results: List[Tuple[Document, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self.documents[idx], score))
        return results


# Backward compatibility & descriptive alias
BM25SparseRetriever = BM25Retriever

