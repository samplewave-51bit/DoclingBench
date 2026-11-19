"""DoclingBench: Enterprise Retrieval Benchmark & Messy Document RAG Platform."""
import os
import io
import re
import time
import json
import logging
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.config import (
    RAGSettings,
    BenchmarkPipelineConfig,
    AVAILABLE_EMBEDDING_MODELS,
    AVAILABLE_CHUNKING_STRATEGIES,
    RETRIEVAL_MODES,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_HYBRID_ALPHA,
    SUPPORTED_LLM_PROVIDERS,
    GROQ_MODELS,
    GEMINI_MODELS,
    DATA_DIR,
    BENCHMARK_DATA_DIR,
    ENTERPRISE_SAMPLE_DIR
)
from src.ingestion.multimodal_parser import EnterpriseDocumentParser
from src.chunking.chunking_engine import MultiStrategyChunker
from src.vector_store import EnterpriseVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_retriever import tokenize
from src.rag_engine import EnterpriseRAGEngine, fetch_groq_models
from src.evaluation.benchmark import RetrievalBenchmarkRunner, PipelineBenchmarkSummary
from src.evaluation.faithfulness import FaithfulnessEvaluator
from src.evaluation.failure_analysis import FailureAnalyzer

# Configure Streamlit Page
st.set_page_config(
    page_title="DoclingBench: Enterprise RAG Retrieval Benchmark",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS
st.markdown("""
<style>
    .benchmark-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    .benchmark-sub {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.3rem;
    }
    .faith-badge-good {
        display: inline-block;
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .faith-badge-warn {
        display: inline-block;
        background-color: #FEF3C7;
        color: #B45309;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .table-container {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.8rem;
        background-color: #FFFFFF;
    }
    .chunk-header-pill {
        display: inline-block;
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stExpander table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 0.6rem 0 !important;
        font-size: 0.82rem !important;
        display: block !important;
        overflow-x: auto !important;
        white-space: nowrap !important;
    }
    .stExpander th, .stExpander td {
        padding: 6px 10px !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        text-align: left !important;
    }
    .stExpander th {
        background-color: rgba(128, 128, 128, 0.15) !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Session State Initialization -----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = EnterpriseVectorStore(embedding_model_name=DEFAULT_EMBEDDING_MODEL)

if "hybrid_retriever" not in st.session_state:
    st.session_state.hybrid_retriever = HybridRetriever(
        vector_store=st.session_state.vector_store,
        documents=[]
    )

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = EnterpriseRAGEngine(
        hybrid_retriever=st.session_state.hybrid_retriever,
        settings=RAGSettings()
    )

if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = []

if "indexed_docs_metadata" not in st.session_state:
    st.session_state.indexed_docs_metadata = []

if "benchmark_history" not in st.session_state:
    st.session_state.benchmark_history = {}

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "inspector_results" not in st.session_state:
    st.session_state.inspector_results = None

if "inspector_query_text" not in st.session_state:
    st.session_state.inspector_query_text = "What was the Q3 Cloud Division operating margin percentage in 2024?"


def format_chunk_content(text: str) -> str:
    """Safely format chunk text for Streamlit:
    1. Escapes literal dollar signs so Streamlit KaTeX doesn't misinterpret currency ($M, $1,420) as LaTeX math equations.
    2. Downgrades giant Markdown H1/H2 headings to bold text so cards don't have oversized H1 fonts.
    3. Cleans isolated horizontal rules (---) without touching table formatting (:---).
    4. Normalizes table rows and boundaries so Markdown tables render cleanly in Streamlit.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # 1. Escape literal dollar signs that cause KaTeX math mode runaway ($M -> \$M, $4,850M -> \$4,850M)
    cleaned = re.sub(r'\\\$', '$', cleaned)
    cleaned = re.sub(r'\$', r'\\$', cleaned)

    # 2. Downgrade Markdown H1/H2/H3 headings so expander contents don't blow up with huge text
    cleaned = re.sub(r'^#\s+(.+)$', r'**\1**\n', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^##\s+(.+)$', r'**\1**\n', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^###\s+(.+)$', r'*\1*\n', cleaned, flags=re.MULTILINE)

    # 3. Clean section dividers (only isolated '---', not markdown table delimiters like ':---')
    cleaned = re.sub(r'(?<![:\-])\s+---\s+(?![:\-])', '\n\n---\n\n', cleaned)

    # 4. If table rows are joined horizontally like `| cell | | next row |`, break them into newlines
    cleaned = re.sub(r'\|\s{1,2}\|', '|\n|', cleaned)

    # 5. Ensure newline before start of table
    cleaned = re.sub(r'([^\n])\s*(\|(?:\s*[^|\n]+\s*\|){2,})', r'\1\n\n\2', cleaned)

    # 6. Ensure newline after end of table row before prose text
    cleaned = re.sub(r'(\|\s*)\s{2,}([A-Z][a-z])', r'\1\n\n\2', cleaned)

    return cleaned


def load_golden_dataset():
    """Load the enterprise benchmark dataset."""
    path = BENCHMARK_DATA_DIR / "enterprise_golden_benchmark.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def index_sample_documents():
    """Parse and index the curated messy enterprise documents with Docling."""
    parser = EnterpriseDocumentParser(parser_engine="docling")
    chunker = MultiStrategyChunker()
    all_chunks = []
    metadata_list = []

    sample_files = list(ENTERPRISE_SAMPLE_DIR.glob("*.pdf")) + \
                   list(ENTERPRISE_SAMPLE_DIR.glob("*.eml")) + \
                   list(ENTERPRISE_SAMPLE_DIR.glob("*.pptx"))

    if not sample_files:
        # Check markdown fallbacks
        sample_files = list(ENTERPRISE_SAMPLE_DIR.glob("*.md")) + list(ENTERPRISE_SAMPLE_DIR.glob("*.eml"))

    progress = st.progress(0, text="Ingesting enterprise documents...")
    for idx, f_path in enumerate(sample_files):
        try:
            parsed_pages = parser.parse_file(f_path, f_path.name)
            chunks = chunker.chunk_documents(parsed_pages, strategy="docling_structural")
            all_chunks.extend(chunks)
            st.session_state.vector_store.add_documents(chunks, doc_name=f_path.name, total_pages=len(parsed_pages))
            metadata_list.append({
                "doc_name": f_path.name,
                "chunks": len(chunks),
                "type": parsed_pages[0].metadata.get("doc_type", "Document") if parsed_pages else "Document",
                "tables_detected": sum(1 for c in chunks if c.metadata.get("is_table", False))
            })
        except Exception as e:
            st.warning(f"Error parsing {f_path.name}: {e}")
        progress.progress((idx + 1) / max(len(sample_files), 1))

    progress.empty()
    st.session_state.indexed_chunks = all_chunks
    st.session_state.indexed_docs_metadata = metadata_list
    st.session_state.hybrid_retriever.update_documents(all_chunks)
    st.session_state.rag_engine.hybrid_retriever = st.session_state.hybrid_retriever


# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### 🔬 System Configuration")

    st.markdown("#### 1. Ingestion & Benchmark Suite")
    if st.button("⚡ Index Enterprise Benchmark Docs", type="primary", use_container_width=True):
        with st.spinner("Ingesting multi-modal enterprise docs with Docling..."):
            index_sample_documents()
            st.success(f"Indexed {len(st.session_state.indexed_chunks)} chunks across {len(st.session_state.indexed_docs_metadata)} documents!")
            st.rerun()

    total_c = len(st.session_state.indexed_chunks)
    total_d = len(st.session_state.indexed_docs_metadata)
    table_c = sum(1 for c in st.session_state.indexed_chunks if c.metadata.get("is_table", False))

    st.caption(f"**Index Status:** `{total_d}` Docs | `{total_c}` Chunks | `{table_c}` Table Chunks")
    st.divider()

    st.markdown("#### 2. Retrieval Parameters")
    retrieval_mode = st.selectbox(
        "Active Retrieval Mode",
        options=RETRIEVAL_MODES,
        index=RETRIEVAL_MODES.index(DEFAULT_RETRIEVAL_MODE),
        help="Choose between Pure Dense, Pure BM25, Hybrid RRF, or Hybrid + Reranker"
    )

    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        options=AVAILABLE_CHUNKING_STRATEGIES,
        index=0,
        help="Compare layout-aware Docling chunking vs naive fixed-size windows"
    )

    top_k = st.slider("Top-K Citations", min_value=1, max_value=10, value=5)
    hybrid_alpha = st.slider("Dense Weight (α) for Weighted Hybrid", min_value=0.0, max_value=1.0, value=DEFAULT_HYBRID_ALPHA, step=0.05)

    st.divider()
    st.markdown("#### 3. LLM Reasoning Engine")
    llm_provider = st.selectbox("LLM Provider", options=SUPPORTED_LLM_PROVIDERS, index=0)

    api_key = ""
    llm_model = "llama-3.1-8b-instant"
    if "Groq" in llm_provider:
        api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
        llm_model = st.selectbox("LLaMA-3 Model", options=GROQ_MODELS, index=0)
    elif "Gemini" in llm_provider:
        api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        llm_model = st.selectbox("Gemini Model", options=GEMINI_MODELS, index=0)

    # Sync settings with RAG engine
    st.session_state.rag_engine.settings.llm_provider = llm_provider
    st.session_state.rag_engine.settings.llm_model = llm_model
    st.session_state.rag_engine.settings.api_key = api_key
    st.session_state.rag_engine.settings.top_k = top_k


# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="benchmark-title">🔬 DoclingBench: Enterprise RAG Retrieval Benchmark</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="benchmark-sub">Empirical evaluation of <b>IBM Docling layout parsing</b>, chunking strategies, embedding representations, <b>BM25+Dense hybrid fusion (RRF)</b>, and <b>cross-encoder reranking</b> on messy enterprise data.</div>',
    unsafe_allow_html=True
)

tab_benchmark, tab_inspector, tab_failure, tab_rag, tab_studio = st.tabs([
    "📊 Benchmark Matrix & Leaderboard",
    "🔍 Interactive Retrieval Inspector",
    "🔬 Failure Analysis & Error Taxonomy",
    "💬 Enterprise RAG Assistant",
    "📁 Document Studio"
])


# =========================================================================
# TAB 1: BENCHMARK MATRIX & LEADERBOARD
# =========================================================================
with tab_benchmark:
    st.subheader("Automated Retrieval Evaluation Matrix")
    st.markdown("""
    Evaluate candidate retrieval architectures against the **Enterprise Golden Benchmark** (tables, financial statements, SLA clauses, exact identifiers, and incident threads).
    """)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_bench_clicked = st.button("🚀 Run Multi-Pipeline Benchmark", type="primary", use_container_width=True)

    with col_info:
        st.info("Benchmarks 4 distinct architectures: **Naive Baseline**, **Docling Dense**, **Docling BM25**, **Docling Hybrid RRF**, and **Docling Hybrid + Reranker**.")

    golden_qa = load_golden_dataset()

    if run_bench_clicked:
        if not st.session_state.indexed_chunks:
            st.warning("Please click 'Index Enterprise Benchmark Docs' in the sidebar first!")
        else:
            runner = RetrievalBenchmarkRunner(golden_qa)
            retriever = st.session_state.hybrid_retriever

            configs = [
                ("1. Naive Baseline (Recursive Chunker + Pure Dense)", "dense_only"),
                ("2. Docling + Pure BM25 Lexical", "sparse_only"),
                ("3. Docling + Pure Dense (FAISS)", "dense_only"),
                ("4. Docling + Hybrid BM25+Dense (RRF)", "hybrid_rrf"),
                ("5. Docling + Hybrid RRF + Cross-Encoder Reranker", "hybrid_rrf_reranked")
            ]

            summaries = []
            bench_bar = st.progress(0, text="Benchmarking pipelines...")
            for idx, (p_name, mode) in enumerate(configs):
                def retrieve_func(q, top_k=10, m=mode):
                    return retriever.retrieve(q, mode=m, top_k=top_k)

                summary = runner.run_benchmark(pipeline_name=p_name, retrieve_fn=retrieve_func)
                summaries.append(summary)
                st.session_state.benchmark_history[p_name] = summary
                bench_bar.progress((idx + 1) / len(configs))

            bench_bar.empty()
            st.success("Benchmark completed across all candidate pipelines!")

    if st.session_state.benchmark_history:
        # Display Leaderboard Table
        summary_rows = []
        for name, summ in st.session_state.benchmark_history.items():
            summary_rows.append({
                "Pipeline Architecture": name,
                "Recall@1": f"{summ.recall_at_1 * 100:.1f}%",
                "Recall@3": f"{summ.recall_at_3 * 100:.1f}%",
                "Recall@5": f"{summ.recall_at_5 * 100:.1f}%",
                "Recall@10": f"{summ.recall_at_10 * 100:.1f}%",
                "MRR": f"{summ.mrr:.3f}",
                "NDCG@5": f"{summ.ndcg_at_5:.3f}",
                "Avg Latency (ms)": f"{summ.avg_latency_ms:.1f}"
            })

        df_leaderboard = pd.DataFrame(summary_rows)
        st.markdown("### 🏆 Benchmark Leaderboard")
        st.dataframe(df_leaderboard, use_container_width=True, hide_index=True)

        # Plot Recall@k Comparison
        st.markdown("### 📈 Recall@K Progression Across Architectures")
        plot_data = []
        for name, s in st.session_state.benchmark_history.items():
            short_name = name.split("(")[0].strip()
            plot_data.append({"Pipeline": short_name, "K": "Recall@1", "Recall": s.recall_at_1})
            plot_data.append({"Pipeline": short_name, "K": "Recall@3", "Recall": s.recall_at_3})
            plot_data.append({"Pipeline": short_name, "K": "Recall@5", "Recall": s.recall_at_5})
            plot_data.append({"Pipeline": short_name, "K": "Recall@10", "Recall": s.recall_at_10})

        df_plot = pd.DataFrame(plot_data)
        st.bar_chart(df_plot, x="K", y="Recall", color="Pipeline")

        # Per Category Insights
        st.markdown("### 🎯 Per-Category Recall@5 Breakdown")
        cat_rows = []
        for name, s in st.session_state.benchmark_history.items():
            row = {"Pipeline": name.split("(")[0].strip()}
            for cat, rec in s.category_recall.items():
                row[cat] = f"{rec * 100:.0f}%"
            cat_rows.append(row)
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Run Multi-Pipeline Benchmark' above to evaluate retrieval performance and view metrics.")


# =========================================================================
# TAB 2: INTERACTIVE RETRIEVAL INSPECTOR
# =========================================================================
with tab_inspector:
    st.subheader("Side-by-Side Retrieval Inspector")
    st.markdown("Compare what **Pure Dense (FAISS)**, **BM25 Lexical**, and **Hybrid RRF + Reranker** return for custom or benchmark queries.")

    def on_inspector_preset_change():
        preset = st.session_state.get("inspector_preset_select")
        if preset and preset != "-- Custom Query --":
            st.session_state["inspector_query_text"] = preset

    qa_presets = [item["query"] for item in golden_qa] if golden_qa else []
    selected_preset = st.selectbox(
        "Select a benchmark query or enter your own below:",
        options=["-- Custom Query --"] + qa_presets,
        key="inspector_preset_select",
        on_change=on_inspector_preset_change
    )

    custom_q = st.text_input(
        "Search Query:",
        key="inspector_query_text"
    )

    c_btn, c_spacer = st.columns([2, 5])
    with c_btn:
        run_inspection = st.button("Inspect Retrieval Differences", type="primary", use_container_width=True)

    if run_inspection:
        if not st.session_state.indexed_chunks:
            st.warning("Please index documents in the sidebar first!")
        else:
            retriever = st.session_state.hybrid_retriever
            
            with st.spinner("Executing retrieval across Dense, BM25, and Hybrid Reranker..."):
                t0 = time.perf_counter()
                res_dense = retriever.search_dense(custom_q, top_k=3)
                t_dense = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                res_sparse = retriever.search_sparse(custom_q, top_k=3)
                t_sparse = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                res_hybrid = retriever.search_hybrid_reranked(custom_q, top_k=3)
                t_hybrid = (time.perf_counter() - t0) * 1000

            st.session_state.inspector_results = {
                "query": custom_q,
                "dense": (res_dense, t_dense),
                "sparse": (res_sparse, t_sparse),
                "hybrid": (res_hybrid, t_hybrid)
            }

    if st.session_state.inspector_results:
        insp = st.session_state.inspector_results
        eval_q = insp.get("query", "")

        # Status Banner explaining the active query
        if custom_q.strip() != eval_q.strip():
            st.warning(f"⚠️ **Query input was changed to:** *\"{custom_q}\"*\n\nResults below reflect previous query: *\"{eval_q}\"*. Click **'Inspect Retrieval Differences'** above to compute fresh rankings.")
        else:
            st.success(f"🔎 **Active Retrieval Results for Query:** *\"{eval_q}\"*")

        q_tokens = tokenize(eval_q)

        def render_inspector_column(title, subtitle, results, latency_ms, metric_name, engine_type="dense"):
            st.markdown(f"#### {title}")
            st.caption(f"⏱️ `{latency_ms:.1f} ms` | {metric_name}")
            
            if engine_type == "sparse":
                token_str = ", ".join([f"`{t}`" for t in q_tokens[:6]]) if q_tokens else "None"
                st.markdown(f"<div style='font-size:0.75rem; color:#64748B; margin-bottom: 0.4rem;'>Extracted Tokens: {token_str}</div>", unsafe_allow_html=True)
            elif engine_type == "hybrid":
                st.markdown("<div style='font-size:0.75rem; color:#64748B; margin-bottom: 0.4rem;'>RRF Fusion (20 pool) → Cross-Encoder Re-ranked</div>", unsafe_allow_html=True)
            elif engine_type == "dense":
                st.markdown("<div style='font-size:0.75rem; color:#64748B; margin-bottom: 0.4rem;'>Semantic Vectors (`all-MiniLM-L6-v2`)</div>", unsafe_allow_html=True)

            if not results:
                st.info("No matching chunks retrieved.")
                return

            for rank, (doc, sc) in enumerate(results, start=1):
                doc_name = doc.metadata.get("doc_name", "Document")
                sec_raw = doc.metadata.get("section_title") or f"Page {doc.metadata.get('page', 1)}"
                is_table = doc.metadata.get("is_table", False) or "|---" in doc.page_content
                table_tag = " | 📊 Table" if is_table else ""
                
                with st.expander(f"Rank {rank}: {doc_name} ({sc:.3f})", expanded=(rank == 1)):
                    st.markdown(f'<div class="chunk-header-pill">📑 Section: {sec_raw}{table_tag}</div>', unsafe_allow_html=True)
                    
                    v_formatted, v_raw = st.tabs(["📄 Formatted View", "🔍 Raw Markdown Text"])
                    with v_formatted:
                        st.markdown(format_chunk_content(doc.page_content))
                    with v_raw:
                        st.code(doc.page_content, language="markdown")

        c_dense, c_sparse, c_hybrid = st.columns(3)
        with c_dense:
            res_d, t_d = insp["dense"]
            render_inspector_column("1. Pure Dense (FAISS)", "Semantic Search", res_d, t_d, "Cosine Similarity", "dense")

        with c_sparse:
            res_s, t_s = insp["sparse"]
            render_inspector_column("2. Pure BM25 (Lexical)", "Keyword Search", res_s, t_s, "BM25 Score", "sparse")

        with c_hybrid:
            res_h, t_h = insp["hybrid"]
            render_inspector_column("3. Hybrid RRF + Reranker", "State of the Art", res_h, t_h, "Cross-Encoder Score", "hybrid")


# =========================================================================
# TAB 3: FAILURE ANALYSIS & ERROR TAXONOMY
# =========================================================================
with tab_failure:
    st.subheader("Systematic Failure Analysis & Root Cause Taxonomy")
    st.markdown("""
    RAG pipelines fail for predictable empirical reasons on messy enterprise documents. 
    Below is a diagnostic classification of failure modes observed across fixed vs layout-aware chunking and dense vs hybrid search.
    """)

    analyzer = FailureAnalyzer()

    # If benchmark ran, analyze its errors; otherwise show standard baseline diagnostic
    active_summary = None
    if st.session_state.benchmark_history:
        active_summary = list(st.session_state.benchmark_history.values())[0]  # Baseline
    elif golden_qa:
        # Pre-seed failure cases for demonstration
        fake_runner = RetrievalBenchmarkRunner(golden_qa)
        if st.session_state.indexed_chunks:
            active_summary = fake_runner.run_benchmark("Baseline Naive Dense", lambda q, top_k: st.session_state.hybrid_retriever.search_dense(q, top_k=top_k))

    if active_summary:
        failures = analyzer.analyze_pipeline_failures(active_summary)

        st.markdown(f"### 🔍 Detected Failures: `{len(failures)}` Missed Queries in Top-5")

        # Taxonomy metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Table Fragmentation", sum(1 for f in failures if f.failure_type == "TABLE_FRAGMENTATION"))
        c2.metric("Vocabulary / Jargon Miss", sum(1 for f in failures if f.failure_type == "VOCABULARY_MISMATCH"))
        c3.metric("Reranker Demotions", sum(1 for f in failures if f.failure_type == "RERANKER_DROP"))
        c4.metric("OCR / Scanned Noise", sum(1 for f in failures if f.failure_type == "OCR_EXTRACTION_ARTIFACT"))

        st.divider()
        st.markdown("### 📋 Failure Case Studies & Corrective Prescriptions")
        for f in failures:
            with st.expander(f"❌ [{f.failure_type}] {f.query} (ID: {f.query_id})"):
                st.markdown(f"**Category:** `{f.category}`")
                st.markdown(f"**Root Cause Diagnosis:** {f.explanation}")
                st.markdown(f"**Top Retrieved Snippet:**\n> *\"{format_chunk_content(f.top_retrieved_snippet)}...\"*")
                st.markdown(f"**Expected Ground Truth Answer:** {f.ground_truth_answer}")
                st.markdown(f"**💡 Engineering Remedy:** `{f.suggested_remedy}`")

        # Download Report Button
        report_md = analyzer.generate_markdown_report(active_summary, failures)
        st.download_button(
            label="📥 Download Full Failure Analysis Report (Markdown)",
            data=report_md,
            file_name="enterprise_rag_failure_analysis_report.md",
            mime="text/markdown"
        )
    else:
        st.info("Index documents and run the benchmark to generate live failure analysis.")


# =========================================================================
# TAB 4: ENTERPRISE GROUNDED RAG ASSISTANT
# =========================================================================
with tab_rag:
    st.subheader("Enterprise Question Answering with Grounding Citations")
    st.markdown("Ask natural language questions against the multi-modal document index with real-time **Answer Faithfulness** scoring.")

    # Render Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("faithfulness"):
                f = msg["faithfulness"]
                f_pct = f.faithfulness_score * 100
                badge_class = "faith-badge-good" if f_pct >= 75 else "faith-badge-warn"
                st.markdown(f'<span class="{badge_class}">🛡️ Faithfulness: {f_pct:.0f}% ({f.supported_claims}/{f.total_claims} claims grounded)</span>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📚 Source Citations & Matched Excerpts"):
                    for s in msg["sources"]:
                        st.markdown(f"- **[{s['citation']}]** (Score: `{s['score']}`)")
                        st.markdown(format_chunk_content(s['snippet']))

    user_query = st.chat_input("Ask a question about the financial tables, emails, contracts, or slides...")
    if user_query:
        # Display user query
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_messages.append({"role": "user", "content": user_query})

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing grounded answer and scoring faithfulness..."):
                response = st.session_state.rag_engine.answer_query(
                    query=user_query,
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    alpha=hybrid_alpha,
                    chat_history=st.session_state.chat_messages
                )

                st.markdown(response["answer"])

                f = response.get("faithfulness")
                if f:
                    f_pct = f.faithfulness_score * 100
                    badge_class = "faith-badge-good" if f_pct >= 75 else "faith-badge-warn"
                    st.markdown(f'<span class="{badge_class}">🛡️ Faithfulness: {f_pct:.0f}% ({f.supported_claims}/{f.total_claims} claims grounded)</span>', unsafe_allow_html=True)
                    if f.unsupported_claims and f_pct < 100:
                        with st.expander("⚠️ Ungrounded Claims Flagged"):
                            for u in f.unsupported_claims:
                                st.markdown(f"- *{u}*")

                if response["sources"]:
                    with st.expander("📚 Source Citations & Matched Excerpts"):
                        for s in response["sources"]:
                            st.markdown(f"- **[{s['citation']}]** (Score: `{s['score']}`)")
                            st.markdown(f"> *{s['snippet']}*")

                st.caption(f"⏱️ Retrieval: `{response['retrieval_latency_ms']} ms` | Generation: `{response['generation_latency_ms']} ms`")

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "faithfulness": f,
                    "sources": response["sources"]
                })


# =========================================================================
# TAB 5: DOCUMENT STUDIO
# =========================================================================
with tab_studio:
    st.subheader("Document Ingestion Studio")
    st.markdown("Upload custom enterprise documents (`.pdf`, `.eml`, `.msg`, `.pptx`, `.docx`, `.txt`) and compare ingestion engines.")

    col_up, col_preview = st.columns([1, 1])

    with col_up:
        uploaded_files = st.file_uploader(
            "Upload Enterprise Documents",
            type=["pdf", "eml", "msg", "pptx", "docx", "txt", "md"],
            accept_multiple_files=True
        )

        parser_choice = st.radio(
            "Ingestion Engine",
            options=["IBM Docling (Deep Layout & Table Structure)", "PyMuPDF (Naive Text Extractor)"],
            index=0,
            help="Compare structured table preservation vs naive text extraction"
        )

        selected_chunker = st.selectbox(
            "Chunking Strategy for Upload",
            options=AVAILABLE_CHUNKING_STRATEGIES,
            index=0
        )

        if st.button("Index Uploaded Files", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please select at least one file to upload.")
            else:
                p_engine = "docling" if "Docling" in parser_choice else "naive_pymupdf"
                u_parser = EnterpriseDocumentParser(parser_engine=p_engine)
                u_chunker = MultiStrategyChunker()

                added_count = 0
                for u_file in uploaded_files:
                    p_docs = u_parser.parse_file(u_file, u_file.name)
                    u_chunks = u_chunker.chunk_documents(p_docs, strategy=selected_chunker)
                    st.session_state.vector_store.add_documents(u_chunks, doc_name=u_file.name)
                    st.session_state.indexed_chunks.extend(u_chunks)
                    added_count += len(u_chunks)

                st.session_state.hybrid_retriever.update_documents(st.session_state.indexed_chunks)
                st.success(f"Successfully processed and indexed {added_count} chunks!")
                st.rerun()

    with col_preview:
        st.markdown("#### Currently Indexed Enterprise Documents")
        if st.session_state.indexed_docs_metadata:
            st.dataframe(pd.DataFrame(st.session_state.indexed_docs_metadata), use_container_width=True, hide_index=True)
        else:
            st.info("No documents indexed yet. Use the sidebar button to load the benchmark suite or upload above.")
