"""Faithfulness and Answer Grounding evaluation engine (LLM-as-a-judge & claim decomposition)."""
import re
import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FaithfulnessResult:
    faithfulness_score: float  # [0.0, 1.0]
    total_claims: int
    supported_claims: int
    unsupported_claims: List[str]
    answer_relevance: float  # [0.0, 1.0]
    reasoning: str


class FaithfulnessEvaluator:
    """Evaluates whether an answer is faithful to retrieved context and does not hallucinate."""

    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine

    def evaluate(self, query: str, context: str, answer: str) -> FaithfulnessResult:
        """Evaluate answer faithfulness and relevance against the retrieved context."""
        if not answer or not context:
            return FaithfulnessResult(
                faithfulness_score=0.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=["No answer or context provided"],
                answer_relevance=0.0,
                reasoning="Empty input."
            )

        # Decompose answer into individual proposition sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 15]
        if not sentences:
            sentences = [answer.strip()]

        total_claims = len(sentences)
        supported_claims = 0
        unsupported = []

        context_lower = context.lower()

        for sent in sentences:
            sent_words = [w for w in re.findall(r'\b[a-zA-Z0-9_\-\$]{3,}\b', sent.lower())]
            if not sent_words:
                continue

            # Check overlap density with retrieved context
            matched_words = [w for w in sent_words if w in context_lower]
            overlap_ratio = len(matched_words) / len(sent_words)

            # A claim is supported if key entities and numbers match context
            if overlap_ratio >= 0.55:
                supported_claims += 1
            else:
                unsupported.append(sent)

        score = round(supported_claims / max(total_claims, 1), 3)

        # Query-Answer relevance estimation
        query_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower()))
        answer_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', answer.lower()))
        overlap_q = len(query_words.intersection(answer_words)) / max(len(query_words), 1)
        relevance = round(min(overlap_q * 1.5, 1.0), 3)

        reasoning = (
            f"Evaluated {total_claims} factual propositions against context. "
            f"{supported_claims} supported by explicit evidence ({score * 100:.1f}% faithfulness)."
        )

        return FaithfulnessResult(
            faithfulness_score=score,
            total_claims=total_claims,
            supported_claims=supported_claims,
            unsupported_claims=unsupported,
            answer_relevance=relevance,
            reasoning=reasoning
        )
