"""Tests for quantitative evaluation suite, metrics, datasets, and failure analysis."""
import pytest
from src.evaluation.benchmark import (
    RetrievalBenchmarkRunner,
    load_benchmark_dataset,
    PipelineBenchmarkSummary,
    QueryBenchmarkResult
)
from src.evaluation.failure_analysis import FailureAnalyzer
from src.evaluation.faithfulness import FaithfulnessEvaluator


class TestEvaluationSuite:

    def test_benchmark_dataset_loader(self):
        """Verify benchmark datasets load with appropriate query counts."""
        poc_data = load_benchmark_dataset("poc_enterprise")
        assert len(poc_data) >= 10, "Enterprise PoC dataset should have >= 10 queries"

        fb_data = load_benchmark_dataset("financebench")
        assert len(fb_data) >= 15, "FinanceBench slice should have >= 15 queries"

        combined = load_benchmark_dataset("combined")
        assert len(combined) == len(poc_data) + len(fb_data)

    def test_benchmark_metrics_computation(self, sample_retriever, mock_golden_dataset):
        """Verify Recall@K, MRR, and NDCG@5 calculations are mathematically bounded in [0, 1]."""
        runner = RetrievalBenchmarkRunner(mock_golden_dataset)

        def retrieve_fn(q, k=10):
            return sample_retriever.retrieve(q, mode="dense_only", top_k=k)

        summary = runner.run_benchmark(
            pipeline_name="Test_Dense",
            retrieve_fn=retrieve_fn,
            parser_type="docling",
            chunker_strategy="docling_structural",
            retriever_mode="dense_only"
        )

        assert isinstance(summary, PipelineBenchmarkSummary)
        assert summary.total_queries == len(mock_golden_dataset)
        assert 0.0 <= summary.recall_at_1 <= 1.0
        assert 0.0 <= summary.recall_at_3 <= 1.0
        assert 0.0 <= summary.recall_at_5 <= 1.0
        assert 0.0 <= summary.recall_at_10 <= 1.0
        assert 0.0 <= summary.mrr <= 1.0
        assert 0.0 <= summary.ndcg_at_5 <= 1.0
        assert summary.avg_latency_ms >= 0.0

    def test_failure_analysis_taxonomy(self):
        """Verify failure taxonomy categorization for retrieval dropouts."""
        analyzer = FailureAnalyzer()

        # Mock results
        results = [
            QueryBenchmarkResult(
                query_id="Q-FAIL-1",
                query="Find margin for division in Table 3",
                category="Tabular",
                latency_ms=10.0,
                retrieved_chunk_ids=["c1"],
                retrieved_texts=["Unrelated paragraph text"],
                retrieved_scores=[0.1],
                hit_at_1=False,
                hit_at_3=False,
                hit_at_5=False,
                hit_at_10=False,
                first_hit_rank=0,
                reciprocal_rank=0.0,
                ndcg_5=0.0,
                ground_truth_answer="28.4% in Table 3"
            )
        ]

        summary = PipelineBenchmarkSummary(
            pipeline_name="Failing_Pipeline",
            total_queries=1,
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            ndcg_at_5=0.0,
            avg_latency_ms=10.0,
            category_recall={"Tabular": 0.0},
            detailed_results=results
        )

        report = analyzer.analyze(summary)
        assert report.total_failures >= 1
        assert len(report.failure_cases) >= 1
        assert hasattr(report.failure_cases[0], "root_cause_category")

    def test_faithfulness_evaluator_claim_extraction(self):
        """Verify claim extraction heuristic breaks text into factual propositions."""
        evaluator = FaithfulnessEvaluator()
        text = "Cloud revenue grew 24.8% YoY. Operating margin was 28.4%. Sarah Chen deployed HF-8842."
        claims = evaluator._extract_claims_heuristic(text)
        assert len(claims) >= 2
