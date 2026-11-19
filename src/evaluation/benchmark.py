"""Quantitative retrieval benchmarking engine computing Recall@k, MRR, NDCG, Latency, and Single-Factor Ablations."""
import os
import json
import time
import math
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path

from src.config import BENCHMARK_DATA_DIR, POC_BENCHMARK_PATH, FINANCEBENCH_SLICE_PATH

logger = logging.getLogger(__name__)


def load_benchmark_dataset(dataset_key: str = "poc_enterprise") -> List[Dict[str, Any]]:
    """Load benchmark dataset by key: 'poc_enterprise', 'financebench', or 'combined'."""
    data = []
    if dataset_key in ["poc_enterprise", "combined"]:
        poc_file = POC_BENCHMARK_PATH if POC_BENCHMARK_PATH.exists() else (BENCHMARK_DATA_DIR / "enterprise_golden_benchmark.json")
        if poc_file.exists():
            with open(poc_file, "r", encoding="utf-8") as f:
                poc_data = json.load(f)
                for item in poc_data:
                    item["dataset_source"] = "Enterprise PoC Golden Set"
                data.extend(poc_data)

    if dataset_key in ["financebench", "combined"]:
        if FINANCEBENCH_SLICE_PATH.exists():
            with open(FINANCEBENCH_SLICE_PATH, "r", encoding="utf-8") as f:
                fb_data = json.load(f)
                for item in fb_data:
                    item["dataset_source"] = "FinanceBench Evaluation Slice"
                data.extend(fb_data)

    if not data:
        logger.warning(f"No benchmark data found for key '{dataset_key}'. Returning empty list.")
    return data


@dataclass
class QueryBenchmarkResult:
    query_id: str
    query: str
    category: str
    latency_ms: float
    retrieved_chunk_ids: List[str]
    retrieved_texts: List[str]
    retrieved_scores: List[float]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    hit_at_10: bool
    first_hit_rank: int  # 1-indexed, 0 if not found
    reciprocal_rank: float
    ndcg_5: float
    ground_truth_answer: str
    dataset_source: str = "Enterprise PoC Golden Set"


@dataclass
class PipelineBenchmarkSummary:
    pipeline_name: str
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    avg_latency_ms: float
    category_recall: Dict[str, float]
    detailed_results: List[QueryBenchmarkResult]
    # Factor attribution metadata (for orthogonal ablations)
    parser_type: str = "docling"
    chunker_strategy: str = "docling_structural"
    retriever_mode: str = "hybrid_rrf_reranked"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_model: str = "all-MiniLM-L6-v2"
    ablation_factor: str = "Composite"
    dataset_name: str = "Enterprise PoC Golden Set"


class RetrievalBenchmarkRunner:
    """Automated benchmark runner over golden enterprise question-answer datasets."""

    def __init__(self, golden_dataset: List[Dict[str, Any]]):
        self.golden_dataset = golden_dataset

    def run_benchmark(
        self,
        pipeline_name: str,
        retrieve_fn: Callable[[str, int], List[Tuple[Any, float]]],
        parser_type: str = "docling",
        chunker_strategy: str = "docling_structural",
        retriever_mode: str = "hybrid_rrf_reranked",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding_model: str = "all-MiniLM-L6-v2",
        ablation_factor: str = "Composite",
        dataset_name: str = "Enterprise PoC Golden Set"
    ) -> PipelineBenchmarkSummary:
        """Execute all benchmark queries against a candidate retrieval function."""
        results: List[QueryBenchmarkResult] = []
        category_hits: Dict[str, List[bool]] = {}

        for item in self.golden_dataset:
            qid = item["query_id"]
            query = item["query"]
            cat = item.get("category", "General")
            gold_doc = item.get("target_document", "")
            gold_keywords = [k.lower() for k in item.get("gold_chunk_keywords", [])]
            gold_ans = item.get("ground_truth_answer", "")
            d_source = item.get("dataset_source", dataset_name)

            # Execute timed retrieval
            start_t = time.perf_counter()
            retrieved: List[Tuple[Any, float]] = retrieve_fn(query, 10)
            latency_ms = (time.perf_counter() - start_t) * 1000.0

            retrieved_chunk_ids = []
            retrieved_texts = []
            retrieved_scores = []
            hit_ranks = []

            for rank_idx, (doc, score) in enumerate(retrieved, start=1):
                chunk_id = doc.metadata.get("chunk_id", f"c_{rank_idx}")
                content = doc.page_content.lower()
                doc_name = doc.metadata.get("doc_name", "")

                retrieved_chunk_ids.append(chunk_id)
                retrieved_texts.append(doc.page_content)
                retrieved_scores.append(round(score, 4))

                # Check hit criteria: keyword presence + doc match
                doc_matched = not gold_doc or (gold_doc.lower() in doc_name.lower())
                keyword_matched = any(k in content for k in gold_keywords) if gold_keywords else True

                if doc_matched and keyword_matched:
                    hit_ranks.append(rank_idx)

            first_hit = hit_ranks[0] if hit_ranks else 0
            hit_1 = 1 in hit_ranks
            hit_3 = any(r <= 3 for r in hit_ranks)
            hit_5 = any(r <= 5 for r in hit_ranks)
            hit_10 = any(r <= 10 for r in hit_ranks)
            rr = (1.0 / first_hit) if first_hit > 0 else 0.0

            # Calculate NDCG@5
            dcg = 0.0
            idcg = 1.0  # binary single-target hit at rank 1
            for r in hit_ranks:
                if r <= 5:
                    dcg += 1.0 / math.log2(r + 1)
            ndcg = min(dcg / idcg, 1.0)

            # Track by category
            if cat not in category_hits:
                category_hits[cat] = []
            category_hits[cat].append(hit_5)

            results.append(QueryBenchmarkResult(
                query_id=qid,
                query=query,
                category=cat,
                latency_ms=round(latency_ms, 2),
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_texts=retrieved_texts,
                retrieved_scores=retrieved_scores,
                hit_at_1=hit_1,
                hit_at_3=hit_3,
                hit_at_5=hit_5,
                hit_at_10=hit_10,
                first_hit_rank=first_hit,
                reciprocal_rank=round(rr, 4),
                ndcg_5=round(ndcg, 4),
                ground_truth_answer=gold_ans,
                dataset_source=d_source
            ))

        total = max(len(results), 1)
        summary = PipelineBenchmarkSummary(
            pipeline_name=pipeline_name,
            total_queries=len(results),
            recall_at_1=round(sum(1 for r in results if r.hit_at_1) / total, 4),
            recall_at_3=round(sum(1 for r in results if r.hit_at_3) / total, 4),
            recall_at_5=round(sum(1 for r in results if r.hit_at_5) / total, 4),
            recall_at_10=round(sum(1 for r in results if r.hit_at_10) / total, 4),
            mrr=round(sum(r.reciprocal_rank for r in results) / total, 4),
            ndcg_at_5=round(sum(r.ndcg_5 for r in results) / total, 4),
            avg_latency_ms=round(sum(r.latency_ms for r in results) / total, 2),
            category_recall={c: round(sum(hits) / max(len(hits), 1), 4) for c, hits in category_hits.items()},
            detailed_results=results,
            parser_type=parser_type,
            chunker_strategy=chunker_strategy,
            retriever_mode=retriever_mode,
            reranker_model=reranker_model,
            embedding_model=embedding_model,
            ablation_factor=ablation_factor,
            dataset_name=dataset_name
        )
        return summary
