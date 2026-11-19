"""Configuration and settings for the Enterprise Retrieval Benchmark and RAG platform."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BENCHMARK_DATA_DIR = DATA_DIR / "benchmark"
ENTERPRISE_SAMPLE_DIR = DATA_DIR / "enterprise_sample"
INDEX_STORE_DIR = BASE_DIR / "vector_indices"
SAMPLE_DIR = BASE_DIR / "sample_data"

DATA_DIR.mkdir(exist_ok=True)
BENCHMARK_DATA_DIR.mkdir(exist_ok=True)
ENTERPRISE_SAMPLE_DIR.mkdir(exist_ok=True)
INDEX_STORE_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

# ----------------- Benchmark Strategies -----------------
AVAILABLE_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "ibm-granite/granite-embedding-125m-english",
    "ibm-granite/granite-embedding-30m-english",
]

AVAILABLE_CHUNKING_STRATEGIES = [
    "docling_structural",      # IBM Docling markdown/layout-aware structural chunking
    "recursive_character",    # Baseline fixed-size sliding window
    "semantic_boundary",      # Chunking on semantic embedding shifts
    "hierarchical_parent"     # Small leaf chunks linked to larger parent context
]

RETRIEVAL_MODES = [
    "dense_only",             # Pure Vector Search (FAISS)
    "sparse_only",            # Pure BM25 Keyword Search
    "hybrid_rrf",             # Reciprocal Rank Fusion (BM25 + Dense)
    "hybrid_weighted",        # Linear Weighted Fusion (alpha * Dense + (1-alpha) * BM25)
    "hybrid_rrf_reranked"     # Hybrid RRF + Cross-Encoder Reranker
]

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNKING_STRATEGY = "docling_structural"
DEFAULT_RETRIEVAL_MODE = "hybrid_rrf_reranked"

# Chunking defaults
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
MIN_TEXT_CHARS_FOR_OCR = 40  # Trigger OCR if page text length is below this
RRF_K_CONSTANT = 60
DEFAULT_HYBRID_ALPHA = 0.6  # 0.6 Dense, 0.4 Sparse BM25

# Evaluation Benchmark Datasets
BENCHMARK_DATASETS = {
    "poc_enterprise": "Enterprise PoC Golden Set (12 queries)",
    "financebench": "FinanceBench Evaluation Slice (20 queries)",
    "combined": "Combined Multi-Modal Suite (32 queries)"
}
POC_BENCHMARK_PATH = BENCHMARK_DATA_DIR / "enterprise_golden_benchmark.json"
FINANCEBENCH_SLICE_PATH = BENCHMARK_DATA_DIR / "financebench_eval_slice.json"

# Evaluation & LLM Provider settings
SUPPORTED_LLM_PROVIDERS = [
    "Groq (LLaMA-3)",
    "Google Gemini",
    "IBM watsonx (Granite)",
    "Local Mock"
]

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768"
]

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

WATSONX_MODELS = [
    "ibm/granite-3-8b-instruct",
    "ibm/granite-3-2b-instruct",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-20b-multilingual"
]
DEFAULT_WATSONX_MODEL = "ibm/granite-3-8b-instruct"
DEFAULT_WATSONX_URL = "https://us-south.ml.cloud.ibm.com"

@dataclass
class BenchmarkPipelineConfig:
    """Configures one specific permutation in the retrieval benchmark."""
    name: str = "Hybrid_RRF_Docling"
    parser_type: str = "docling"           # "docling" or "naive_pymupdf"
    chunking_strategy: str = "docling_structural"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    retrieval_mode: str = "hybrid_rrf_reranked"
    top_k: int = 5
    candidate_pool_k: int = 20
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA
    reranker_model: str = DEFAULT_RERANKER_MODEL
    llm_provider: str = "Groq (LLaMA-3)"
    llm_model: str = "llama-3.1-8b-instant"
    api_key: str = ""
    watsonx_project_id: str = ""
    watsonx_url: str = DEFAULT_WATSONX_URL
    temperature: float = 0.2

# Backward compatibility alias
RAGSettings = BenchmarkPipelineConfig
