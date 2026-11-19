"""Comprehensive tests for Enterprise Retrieval Benchmark, Ingestion, and Hybrid Retrieval."""
import unittest
import json
from pathlib import Path

from src.ingestion.multimodal_parser import EnterpriseDocumentParser
from src.chunking.chunking_engine import MultiStrategyChunker
from src.vector_store import EnterpriseVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.benchmark import RetrievalBenchmarkRunner
from src.evaluation.faithfulness import FaithfulnessEvaluator
from src.evaluation.failure_analysis import FailureAnalyzer
from src.config import BENCHMARK_DATA_DIR, ENTERPRISE_SAMPLE_DIR

class TestEnterpriseRetrievalBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = EnterpriseDocumentParser(parser_engine="docling")
        cls.chunker = MultiStrategyChunker()
        cls.vector_store = EnterpriseVectorStore(embedding_model_name="all-MiniLM-L6-v2")

        # Load golden dataset
        gold_file = BENCHMARK_DATA_DIR / "enterprise_golden_benchmark.json"
        with open(gold_file, "r", encoding="utf-8") as f:
            cls.golden_data = json.load(f)

    def test_01_email_parsing(self):
        eml_file = ENTERPRISE_SAMPLE_DIR / "Incident_WarRoom_P1_Outage.eml"
        self.assertTrue(eml_file.exists())
        docs = self.parser.parse_file(eml_file, eml_file.name)
        self.assertGreater(len(docs), 0)
        self.assertEqual(docs[0].metadata["doc_type"], "Email")
        self.assertIn("INC-7721-P1", docs[0].page_content)
        self.assertIn("Sarah Chen", docs[0].page_content)

    def test_02_pdf_table_ingestion(self):
        pdf_file = ENTERPRISE_SAMPLE_DIR / "Q3_Global_Enterprise_Financial_Report.pdf"
        self.assertTrue(pdf_file.exists())
        docs = self.parser.parse_file(pdf_file, pdf_file.name)
        self.assertGreater(len(docs), 0)
        self.assertIn("Operating Segment", docs[0].page_content)

    def test_03_chunking_strategies(self):
        pdf_file = ENTERPRISE_SAMPLE_DIR / "Q3_Global_Enterprise_Financial_Report.pdf"
        docs = self.parser.parse_file(pdf_file, pdf_file.name)

        strategies = ["docling_structural", "recursive_character", "semantic_boundary", "hierarchical_parent"]
        for strat in strategies:
            chunks = self.chunker.chunk_documents(docs, strategy=strat)
            self.assertGreater(len(chunks), 0)
            self.assertEqual(chunks[0].metadata["chunking_strategy"], strat)

    def test_04_hybrid_retrieval(self):
        # Index sample docs
        eml_file = ENTERPRISE_SAMPLE_DIR / "Incident_WarRoom_P1_Outage.eml"
        fin_file = ENTERPRISE_SAMPLE_DIR / "Q3_Global_Enterprise_Financial_Report.pdf"

        docs = self.parser.parse_file(eml_file, eml_file.name) + self.parser.parse_file(fin_file, fin_file.name)
        chunks = self.chunker.chunk_documents(docs, strategy="docling_structural")

        self.vector_store.clear()
        self.vector_store.add_documents(chunks, doc_name="test_corpus")
        hybrid = HybridRetriever(vector_store=self.vector_store, documents=chunks)

        # 1. Test Dense
        dense_hits = hybrid.search_dense("Cloud Infrastructure margin", top_k=3)
        self.assertGreater(len(dense_hits), 0)

        # 2. Test BM25
        bm25_hits = hybrid.search_sparse("hotfix HF-8842", top_k=3)
        self.assertGreater(len(bm25_hits), 0)
        self.assertIn("HF-8842", bm25_hits[0][0].page_content)

        # 3. Test RRF
        rrf_hits = hybrid.search_hybrid_rrf("operating margin 28.4%", top_k=3)
        self.assertGreater(len(rrf_hits), 0)

    def test_05_benchmark_runner(self):
        eml_file = ENTERPRISE_SAMPLE_DIR / "Incident_WarRoom_P1_Outage.eml"
        docs = self.parser.parse_file(eml_file, eml_file.name)
        chunks = self.chunker.chunk_documents(docs, strategy="docling_structural")

        self.vector_store.clear()
        self.vector_store.add_documents(chunks, doc_name="Incident_WarRoom_P1_Outage.eml")
        hybrid = HybridRetriever(vector_store=self.vector_store, documents=chunks)

        runner = RetrievalBenchmarkRunner(self.golden_data[:4])
        summary = runner.run_benchmark("Test RRF", lambda q, top_k: hybrid.search_hybrid_rrf(q, top_k=top_k))

        self.assertIsNotNone(summary)
        self.assertGreaterEqual(summary.recall_at_5, 0.0)
        self.assertGreaterEqual(summary.mrr, 0.0)

    def test_06_failure_analysis(self):
        runner = RetrievalBenchmarkRunner(self.golden_data[:4])
        # Function that returns empty list to trigger failure detection
        summary = runner.run_benchmark("Empty Pipeline", lambda q, top_k: [])
        analyzer = FailureAnalyzer()
        failures = analyzer.analyze_pipeline_failures(summary)
        self.assertEqual(len(failures), len(summary.detailed_results))
        report = analyzer.generate_markdown_report(summary, failures)
        self.assertIn("Error Taxonomy Distribution", report)

    def test_07_faithfulness_evaluation(self):
        evaluator = FaithfulnessEvaluator()
        res = evaluator.evaluate(
            query="What is the operating margin?",
            context="The cloud division operating margin is 28.4% in 2024.",
            answer="The cloud division operating margin reached 28.4% in 2024."
        )
        self.assertEqual(res.faithfulness_score, 1.0)
        self.assertGreater(res.answer_relevance, 0.5)

if __name__ == "__main__":
    unittest.main()
