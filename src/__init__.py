"""Enterprise Retrieval Benchmark & Messy Document RAG Platform."""
__version__ = "2.0.0"

from src.rag_engine import EnterpriseRAGEngine, RAGEngine, fetch_groq_models
from src.ingestion.multimodal_parser import EnterpriseDocumentParser
from src.chunking.chunking_engine import MultiStrategyChunker
from src.vector_store import EnterpriseVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.benchmark import RetrievalBenchmarkRunner
from src.evaluation.faithfulness import FaithfulnessEvaluator
from src.evaluation.failure_analysis import FailureAnalyzer

__all__ = [
    "EnterpriseRAGEngine",
    "RAGEngine",
    "EnterpriseDocumentParser",
    "MultiStrategyChunker",
    "EnterpriseVectorStore",
    "HybridRetriever",
    "RetrievalBenchmarkRunner",
    "FaithfulnessEvaluator",
    "FailureAnalyzer",
    "fetch_groq_models"
]
