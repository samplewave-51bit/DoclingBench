"""Systematic failure analysis and error taxonomy for enterprise RAG pipelines."""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.evaluation.benchmark import QueryBenchmarkResult, PipelineBenchmarkSummary

logger = logging.getLogger(__name__)

ERROR_TAXONOMY = {
    "TABLE_FRAGMENTATION": "Tabular structure was sliced across chunk boundaries, separating headers from data cells.",
    "VOCABULARY_MISMATCH": "Query contained specialized alphanumeric codes, SKUs, or jargon missed by dense embeddings.",
    "RERANKER_DROP": "Relevant chunk was retrieved in the initial candidate pool (top-20) but demoted by the cross-encoder reranker.",
    "LOW_GROUNDING_HALLUCINATION": "Retrieved context was insufficient or conflicting, causing the LLM to hallucinate unsupported claims.",
    "OCR_EXTRACTION_ARTIFACT": "Scanned document or diagram contained noisy, garbled, or low-contrast text."
}

@dataclass
class FailureCase:
    query_id: str
    query: str
    category: str
    failure_type: str
    explanation: str
    retrieved_at_rank: int
    top_retrieved_snippet: str
    ground_truth_answer: str
    suggested_remedy: str

    @property
    def root_cause_category(self) -> str:
        return self.failure_type


@dataclass
class FailureReport:
    total_failures: int
    failure_cases: List[FailureCase]

    def __iter__(self):
        return iter(self.failure_cases)

    def __len__(self):
        return len(self.failure_cases)


class FailureAnalyzer:
    """Diagnoses failure modes across benchmark runs and generates actionable reports."""

    def analyze(self, summary: PipelineBenchmarkSummary, raw_documents: List[Any] = None) -> FailureReport:
        """Alias returning a structured FailureReport."""
        cases = self.analyze_pipeline_failures(summary, raw_documents)
        return FailureReport(total_failures=len(cases), failure_cases=cases)

    def analyze_pipeline_failures(
        self,
        summary: PipelineBenchmarkSummary,
        raw_documents: List[Any] = None
    ) -> List[FailureCase]:
        """Examine query benchmark results and categorize failure causes."""
        failures: List[FailureCase] = []

        for q_res in summary.detailed_results:
            if not q_res.hit_at_5:
                # Determine failure type based on query characteristics
                query_lower = q_res.query.lower()
                cat = q_res.category.lower()

                top_snippet = q_res.retrieved_texts[0][:200] if q_res.retrieved_texts else "None"

                if "table" in cat or "revenue" in query_lower or "ebitda" in query_lower or "financial" in cat:
                    f_type = "TABLE_FRAGMENTATION"
                    explanation = "Query requires multi-column tabular correlation. Fixed-size chunker broke table rows."
                    remedy = "Use IBM Docling structural Markdown parsing to preserve full table schemas intact."
                elif any(c in q_res.query for c in ["-", "_", "INC", "SKU", "PROJ", "API", "CVE", "P1"]):
                    f_type = "VOCABULARY_MISMATCH"
                    explanation = "Bi-encoder semantic embedding failed to match rare exact alphanumeric code/identifier."
                    remedy = "Enable BM25 sparse hybrid retrieval with Reciprocal Rank Fusion (RRF)."
                elif q_res.hit_at_10 and not q_res.hit_at_5:
                    f_type = "RERANKER_DROP"
                    explanation = "Relevant passage was captured in top-10 candidate pool but scored below top-5 cut."
                    remedy = "Fine-tune cross-encoder reranker or increase candidate pool size K to 25."
                elif "scan" in cat or "diagram" in cat:
                    f_type = "OCR_EXTRACTION_ARTIFACT"
                    explanation = "Native text was unavailable; optical recognition noise degraded keyword matching."
                    remedy = "Utilize Docling's integrated EasyOCR / RapidOCR model with contrast normalization."
                else:
                    f_type = "LOW_GROUNDING_HALLUCINATION"
                    explanation = "Context window lacks sufficient direct grounding for the query."
                    remedy = "Expand top-K retrieval and employ hierarchical parent-document retrieval."

                failures.append(FailureCase(
                    query_id=q_res.query_id,
                    query=q_res.query,
                    category=q_res.category,
                    failure_type=f_type,
                    explanation=explanation,
                    retrieved_at_rank=q_res.first_hit_rank,
                    top_retrieved_snippet=top_snippet,
                    ground_truth_answer=q_res.ground_truth_answer,
                    suggested_remedy=remedy
                ))

        return failures

    def generate_markdown_report(self, summary: PipelineBenchmarkSummary, failures: List[FailureCase]) -> str:
        """Generate a GitHub-flavored Markdown failure analysis report."""
        lines = [
            f"# 🔬 Retrieval Failure & Error Taxonomy Report: {summary.pipeline_name}",
            "",
            "## 1. Benchmark Performance Snapshot",
            f"- **Total Benchmark Queries**: {summary.total_queries}",
            f"- **Recall@1**: `{summary.recall_at_1 * 100:.1f}%`",
            f"- **Recall@5**: `{summary.recall_at_5 * 100:.1f}%`",
            f"- **MRR**: `{summary.mrr:.3f}`",
            f"- **NDCG@5**: `{summary.ndcg_at_5:.3f}`",
            f"- **Avg Latency**: `{summary.avg_latency_ms:.1f} ms`",
            f"- **Failed Queries (Missed Top-5)**: `{len(failures)}`",
            "",
            "## 2. Error Taxonomy Distribution",
            "| Failure Category | Occurrences | Percentage | Primary Root Cause |",
            "| :--- | :---: | :---: | :--- |"
        ]

        # Count frequencies
        counts = {}
        for f in failures:
            counts[f.failure_type] = counts.get(f.failure_type, 0) + 1

        for f_type, cnt in counts.items():
            pct = (cnt / max(len(failures), 1)) * 100
            desc = ERROR_TAXONOMY.get(f_type, "")
            lines.append(f"| `{f_type}` | {cnt} | {pct:.1f}% | {desc} |")

        lines.extend([
            "",
            "## 3. Detailed Failure Case Studies",
            ""
        ])

        for f in failures:
            lines.extend([
                f"### Query: `{f.query}` (ID: {f.query_id})",
                f"- **Category**: {f.category}",
                f"- **Error Mode**: `{f.failure_type}`",
                f"- **Diagnosis**: {f.explanation}",
                f"- **First Hit Rank**: {f'Rank {f.retrieved_at_rank}' if f.retrieved_at_rank > 0 else 'Not in Top 10'}",
                f"- **Top Retrieved Chunk Snippet**:\n> *\"{f.top_retrieved_snippet}...\"*",
                f"- **Expected Gold Answer**: {f.ground_truth_answer}",
                f"- **Remediation**: 💡 {f.suggested_remedy}",
                ""
            ])

        return "\n".join(lines)
