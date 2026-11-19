"""Quantitative retrieval benchmarking engine computing Recall@k, MRR, NDCG, and Latency."""
import time
import math
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

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


class RetrievalBenchmarkRunner:
    """Automated benchmark runner over golden enterprise question-answer datasets."""

    def __init__(self, golden_dataset: List[Dict[str, Any]]):
        self.golden_dataset = golden_dataset

    def run_benchmark(self, pipeline_name: str, retrieve_fn) -> PipelineBenchmarkSummary:
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

            # Execute timed retrieval
            start_t = time.perf_counter()
            retrieved: List[Tuple[Any, float]] = retrieve_fn(query, top_k=10)
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
            idcg = 1.0  # single ideal binary hit at rank 1
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
                ground_truth_answer=gold_ans
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
            detailed_results=results
        )
        return summary
