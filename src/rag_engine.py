"""Enterprise RAG engine supporting Hybrid Retrieval (Dense + BM25 + Cross-Encoder) with Live Faithfulness Scoring."""
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.faithfulness import FaithfulnessEvaluator, FaithfulnessResult
from src.config import RAGSettings, DEFAULT_RETRIEVAL_MODE, DEFAULT_HYBRID_ALPHA

logger = logging.getLogger(__name__)

ENTERPRISE_QA_SYSTEM_PROMPT = """You are an expert Enterprise Information Architect and Research Analyst.
Your job is to answer queries based STRICTLY on the provided enterprise documents (financial reports, tables, email threads, contracts, slides, and audits).

Guidelines:
1. FACTUAL GROUNDING: Rely strictly and solely on the provided document excerpts. Never extrapolate, hallucinate, or assume metrics not stated. If the information is not in the context, explicitly state: "The provided context does not contain sufficient information to answer this question."
2. EXACT TABLE & METRIC ACCURACY: When referencing numbers, EBITDA margins, dates, SLA codes, or percentages from tables, quote the exact values and cite the table or section.
3. CITATIONS: Ground every assertion with an inline citation in square brackets referencing the document name and section, e.g. `[Q3_Financial_Report.pdf - Table 2]` or `[Incident_WarRoom.eml - Section 3]`.
4. FORMAT: Use clean bullet points, bold key terms, and markdown tables where appropriate.

Context:
{context}
"""

EXCLUDED_MODEL_KEYWORDS = [
    "canopylabs", "orpheus", "whisper", "guard", "embedding", 
    "tts", "audio", "vision", "moderation", "distil", "allam"
]
ALLOWED_MODEL_FAMILIES = ["llama", "gemma", "mixtral", "qwen", "deepseek"]

PREFERRED_LLAMA_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

_BLACKLISTED_MODELS = set()

def fetch_groq_models(api_key: str) -> List[str]:
    """Fetch live list of chat models available to Groq key."""
    if not api_key:
        return PREFERRED_LLAMA_MODELS
    try:
        from groq import Groq
        client = Groq(api_key=api_key.strip())
        resp = client.models.list()
        chat_models = []
        for m in resp.data:
            mid = m.id.lower()
            if any(k in mid for k in EXCLUDED_MODEL_KEYWORDS):
                continue
            if any(fam in mid for fam in ALLOWED_MODEL_FAMILIES):
                chat_models.append(m.id)

        def model_priority(m_id: str) -> int:
            if m_id in PREFERRED_LLAMA_MODELS:
                return PREFERRED_LLAMA_MODELS.index(m_id)
            elif "llama" in m_id.lower():
                return 50
            return 99

        sorted_models = sorted(chat_models, key=model_priority)
        return sorted_models if sorted_models else PREFERRED_LLAMA_MODELS
    except Exception as e:
        logger.warning(f"Error querying Groq models list: {e}")
        return PREFERRED_LLAMA_MODELS


class MockEnterpriseLLM:
    """Intelligent fallback for demonstration when no external API key is configured."""

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        prompt_text = messages[-1].content if messages else ""
        return AIMessage(
            content=(
                "**[Demo Mode Response - Provide a Groq or Gemini API Key in the sidebar for full LLM reasoning]**\n\n"
                f"**Query Received:**\n> {prompt_text}\n\n"
                "**Context Retrieved from Enterprise Documents:**\n"
                "The hybrid retrieval engine successfully surfaced top matching passages and table cells. "
                "See the 'Retrieved Sources & Evidence Context' panel below for exact matched chunks and similarity scores."
            )
        )

# Backward compatibility alias
MockEducationalLLM = MockEnterpriseLLM



class EnterpriseRAGEngine:
    """Orchestrates document retrieval via Hybrid/Dense/BM25/Reranker, LLM generation, and faithfulness evaluation."""

    def __init__(self, hybrid_retriever: Any, settings: Optional[RAGSettings] = None):
        if not isinstance(hybrid_retriever, HybridRetriever):
            # Auto-wrap if a raw vector store was passed
            self.hybrid_retriever = HybridRetriever(vector_store=hybrid_retriever)
        else:
            self.hybrid_retriever = hybrid_retriever
        self.settings = settings or RAGSettings()
        self.llm = self._init_llm()
        self.evaluator = FaithfulnessEvaluator(llm_engine=self.llm)

    @property
    def vector_store(self):
        """Expose vector store for backward compatibility."""
        if hasattr(self.hybrid_retriever, "vector_store"):
            return self.hybrid_retriever.vector_store
        return self.hybrid_retriever

    def update_settings(self, settings: RAGSettings):
        """Update RAG parameters and reinitialize LLM if provider/key changed."""
        self.settings = settings
        self.llm = self._init_llm()
        self.evaluator = FaithfulnessEvaluator(llm_engine=self.llm)

    def _init_llm(self):
        """Initialize the selected LLM provider via LangChain."""
        provider = self.settings.llm_provider
        logger.info(f"Initializing LLM provider: {provider} with model {self.settings.llm_model}")

        # 1. Groq (LLaMA-3 models)
        if "Groq" in provider:
            groq_key = (self.settings.api_key or os.getenv("GROQ_API_KEY", "")).strip()
            if not groq_key:
                logger.warning("Groq API key not found. Using Mock fallback.")
                return MockEducationalLLM()

            current_model = (self.settings.llm_model or "").strip()
            if any(k in current_model.lower() for k in EXCLUDED_MODEL_KEYWORDS) or current_model in _BLACKLISTED_MODELS:
                self.settings.llm_model = "llama-3.1-8b-instant"

            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    api_key=groq_key,
                    model_name=self.settings.llm_model,
                    temperature=self.settings.temperature,
                    max_tokens=2048
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChatGroq: {e}")
                return MockEducationalLLM()

        # 2. Google Gemini
        elif "Gemini" in provider:
            gemini_key = self.settings.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
            if not gemini_key:
                logger.warning("Gemini API key not found. Using Mock fallback.")
                return MockEducationalLLM()
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    google_api_key=gemini_key,
                    model=self.settings.llm_model,
                    temperature=self.settings.temperature,
                    max_output_tokens=2048
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
                return MockEducationalLLM()

        return MockEducationalLLM()

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Invoke LLM with automatic multi-model fallback."""
        try:
            return self.llm.invoke(messages)
        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()
            recoverable = any(ind in err_lower for ind in ["model_terms_required", "not have access", "404", "400"])

            groq_key = (self.settings.api_key or os.getenv("GROQ_API_KEY", "")).strip()
            if recoverable and "Groq" in self.settings.llm_provider and groq_key:
                failed = self.settings.llm_model
                _BLACKLISTED_MODELS.add(failed)
                candidates = [m for m in PREFERRED_LLAMA_MODELS if m not in _BLACKLISTED_MODELS]
                for cand in candidates:
                    try:
                        from langchain_groq import ChatGroq
                        fb = ChatGroq(api_key=groq_key, model_name=cand, temperature=self.settings.temperature)
                        resp = fb.invoke(messages)
                        self.llm = fb
                        self.settings.llm_model = cand
                        return resp
                    except Exception:
                        _BLACKLISTED_MODELS.add(cand)

            raise e

    def format_retrieved_context(self, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """Format retrieved chunks into a clean context string with citations."""
        if not retrieved_docs:
            return "No relevant enterprise context found."

        context_blocks = []
        for idx, (doc, score) in enumerate(retrieved_docs, start=1):
            meta = doc.metadata
            doc_name = meta.get("doc_name", "Enterprise Document")
            sec_title = meta.get("section_title", f"Page {meta.get('page', 1)}")
            
            header = f"--- [SOURCE {idx}: {doc_name} | Section: {sec_title} | Score: {score:.3f}] ---"
            block = f"{header}\n{doc.page_content.strip()}"
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def answer_query(
        self,
        query: str,
        retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
        top_k: int = 5,
        candidate_pool_k: int = 20,
        alpha: float = DEFAULT_HYBRID_ALPHA,
        chat_history: Optional[List[Dict[str, str]]] = None,
        filter_doc_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Query the knowledge base using hybrid retrieval, synthesize an answer, and score faithfulness."""
        start_t = time.perf_counter()

        # 1. Retrieve relevant chunks using the configured mode
        filter_dict = {"doc_names": filter_doc_names} if filter_doc_names else None
        retrieved_results = self.hybrid_retriever.retrieve(
            query=query,
            mode=retrieval_mode,
            top_k=top_k,
            candidate_pool_k=candidate_pool_k,
            alpha=alpha,
            filter_dict=filter_dict
        )

        retrieval_latency_ms = round((time.perf_counter() - start_t) * 1000.0, 1)

        if not retrieved_results:
            return {
                "answer": "No relevant documents found. Please index documents in the Ingestion Studio.",
                "sources": [],
                "context": "",
                "faithfulness": None,
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": 0.0
            }

        formatted_context = self.format_retrieved_context(retrieved_results)

        # 2. Build Prompt Messages
        messages: List[BaseMessage] = [
            SystemMessage(content=ENTERPRISE_QA_SYSTEM_PROMPT.format(context=formatted_context))
        ]

        if chat_history:
            for turn in chat_history[-4:]:
                role = turn.get("role")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=query))

        # 3. LLM Generation
        gen_start = time.perf_counter()
        try:
            response = self.invoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            answer_text = f"An error occurred while generating the answer: {str(e)}"
        gen_latency_ms = round((time.perf_counter() - gen_start) * 1000.0, 1)

        # 4. Live Faithfulness / Grounding Evaluation
        faithfulness_res = self.evaluator.evaluate(
            query=query,
            context=formatted_context,
            answer=answer_text
        )

        # 5. Format structured sources
        sources_summary = []
        for doc, score in retrieved_results:
            sec_name = doc.metadata.get("section_title") or f"Page {doc.metadata.get('page', 1)}"
            doc_name = doc.metadata.get("doc_name", "Document")
            sources_summary.append({
                "citation": f"{doc_name} ({sec_name})",
                "doc_name": doc_name,
                "section": sec_name,
                "score": round(score, 4),
                "is_table": doc.metadata.get("is_table", False),
                "snippet": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
            })

        return {
            "answer": answer_text,
            "sources": sources_summary,
            "context": formatted_context,
            "faithfulness": faithfulness_res,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": gen_latency_ms,
            "total_latency_ms": round(retrieval_latency_ms + gen_latency_ms, 1)
        }

# Backward compatibility alias
RAGEngine = EnterpriseRAGEngine
