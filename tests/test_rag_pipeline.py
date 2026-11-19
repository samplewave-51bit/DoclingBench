"""Tests for end-to-end RAG pipeline, LLM initialization, watsonx Granite, and mock fallbacks."""
import pytest
from src.rag_engine import EnterpriseRAGEngine, MockEnterpriseLLM
from src.config import RAGSettings


class TestRAGPipeline:

    def test_rag_engine_mock_fallback(self, sample_retriever):
        """When no API keys are provided, engine must gracefully fall back to Mock LLM."""
        settings = RAGSettings(llm_provider="Local Mock")
        engine = EnterpriseRAGEngine(hybrid_retriever=sample_retriever, settings=settings)

        assert isinstance(engine.llm, MockEnterpriseLLM)
        resp = engine.answer("What was the Cloud Infrastructure margin in Q3?")
        assert resp["answer"] is not None
        assert "Demo Mode" in resp["answer"] or "hybrid retrieval engine" in resp["answer"]
        assert len(resp["retrieved_sources"]) >= 1

    def test_watsonx_granite_configuration(self, sample_retriever):
        """Verify IBM watsonx Granite provider switches correctly and handles missing keys safely."""
        settings = RAGSettings(
            llm_provider="IBM watsonx (Granite)",
            llm_model="ibm/granite-3-8b-instruct",
            api_key="",
            watsonx_project_id=""
        )
        engine = EnterpriseRAGEngine(hybrid_retriever=sample_retriever, settings=settings)
        # Without credentials, it should safely return MockEnterpriseLLM indicating watsonx
        assert isinstance(engine.llm, MockEnterpriseLLM)
        assert "watsonx" in engine.llm.provider_name.lower() or "granite" in engine.llm.provider_name.lower()

    def test_citation_context_formatting(self, sample_retriever):
        """Verify context assembly attaches document names and sections."""
        engine = EnterpriseRAGEngine(hybrid_retriever=sample_retriever)
        docs = sample_retriever.search_dense("Cloud Infrastructure", top_k=2)
        formatted = engine._format_context(docs)
        assert "[SOURCE 1:" in formatted
        assert "Score:" in formatted
        assert len(formatted) > 50
