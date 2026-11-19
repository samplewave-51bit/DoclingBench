"""Pytest configuration and shared fixtures for DoclingBench."""
import pytest
import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retrieval.hybrid_retriever import HybridRetriever
from src.vector_store import EnterpriseVectorStore


class MockFastEmbeddings(Embeddings):
    """Deterministic fast embedding fixture for unit testing (avoids loading PyTorch models)."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        # Hash text to deterministic pseudo-vector
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


@pytest.fixture
def mock_embeddings():
    return MockFastEmbeddings(dim=64)


@pytest.fixture
def sample_documents() -> List[Document]:
    return [
        Document(
            page_content=(
                "# Consolidated Financial Performance\n\n"
                "| Operating Segment | Q3 2024 Revenue | Operating Margin |\n"
                "| :--- | :--- | :--- |\n"
                "| Cloud Infrastructure | $1,840M | 28.4% |\n"
                "| Software & SaaS | $920M | 34.1% |\n"
                "| Legacy Hardware | $650M | 12.0% |\n"
            ),
            metadata={
                "chunk_id": "fin_c1",
                "doc_name": "Q3_Global_Enterprise_Financial_Report.pdf",
                "doc_type": "PDF (Docling Structured)",
                "page": 1,
                "is_table": True,
                "chunking_strategy": "docling_structural"
            }
        ),
        Document(
            page_content=(
                "From: incident-lead@enterprise.com\n"
                "To: war-room@enterprise.com\n"
                "Subject: [P1] Major API Degradation INC-7721-P1\n"
                "Date: 2024-09-14 02:15 UTC\n\n"
                "Root cause identified: Database connection pool exhaustion caused by unindexed queries in release v4.12.0. "
                "Assigned hotfix HF-8842 to Sarah Chen with target deployment at 04:30 UTC."
            ),
            metadata={
                "chunk_id": "eml_c1",
                "doc_name": "Incident_WarRoom_P1_Outage.eml",
                "doc_type": "Email",
                "page": 1,
                "is_table": False,
                "chunking_strategy": "docling_structural"
            }
        ),
        Document(
            page_content=(
                "## 4. Service Level Guarantees (SLA)\n"
                "4.1 Uptime Commitment: Provider guarantees 99.95% monthly service availability.\n"
                "4.2 Penalties: If monthly availability falls below 99.0%, a 25% service credit applies.\n"
                "4.3 Disaster Recovery: Guaranteed Recovery Time Objective (RTO) is 15 minutes."
            ),
            metadata={
                "chunk_id": "sla_c1",
                "doc_name": "Enterprise_Cloud_Procurement_SLA.pdf",
                "doc_type": "PDF (Docling Structured)",
                "page": 1,
                "is_table": False,
                "chunking_strategy": "docling_structural"
            }
        )
    ]


@pytest.fixture
def mock_golden_dataset():
    return [
        {
            "query_id": "Q-01",
            "query": "What was the Cloud Infrastructure operating margin in Q3 2024?",
            "category": "Tabular / Financial",
            "target_document": "Q3_Global_Enterprise_Financial_Report.pdf",
            "gold_chunk_keywords": ["cloud infrastructure", "28.4%"],
            "ground_truth_answer": "The Cloud Infrastructure operating margin was 28.4% in Q3 2024."
        },
        {
            "query_id": "Q-02",
            "query": "What root cause caused incident INC-7721-P1?",
            "category": "Email Incident",
            "target_document": "Incident_WarRoom_P1_Outage.eml",
            "gold_chunk_keywords": ["inc-7721-p1", "connection pool exhaustion"],
            "ground_truth_answer": "Database connection pool exhaustion caused by unindexed queries in v4.12.0."
        },
        {
            "query_id": "Q-03",
            "query": "What is the guaranteed RTO in Section 4.3 of the SLA?",
            "category": "SLA Clause",
            "target_document": "Enterprise_Cloud_Procurement_SLA.pdf",
            "gold_chunk_keywords": ["rto", "15 minutes"],
            "ground_truth_answer": "Guaranteed Recovery Time Objective (RTO) is 15 minutes."
        }
    ]


@pytest.fixture
def sample_retriever(sample_documents, mock_embeddings):
    """Instantiate a fast working hybrid retriever with mock embeddings."""
    from langchain_community.vectorstores import FAISS

    faiss_db = FAISS.from_documents(sample_documents, mock_embeddings)
    vstore = EnterpriseVectorStore()
    vstore.embeddings = mock_embeddings
    vstore.faiss_index = faiss_db

    retriever = HybridRetriever(vector_store=vstore, documents=sample_documents)
    return retriever
