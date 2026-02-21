"""
RAG Metrics

RAG quality evaluation metrics.

Based on: src/observability/evaluation.py
"""

from dataclasses import dataclass, field
from typing import Any, Protocol
import re


@dataclass
class RAGScore:
    """RAG evaluation scores."""

    # Retrieval metrics
    context_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Generation metrics
    answer_relevance: float = 0.0
    answer_faithfulness: float = 0.0
    answer_completeness: float = 0.0

    # Overall
    overall_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "context_relevance": self.context_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_relevance": self.answer_relevance,
            "answer_faithfulness": self.answer_faithfulness,
            "answer_completeness": self.answer_completeness,
            "overall_score": self.overall_score,
        }


class LLMProtocol(Protocol):
    """Protocol for LLM judging."""

    async def complete(self, prompt: str) -> str: ...


class RAGMetrics:
    """
    RAG quality evaluation.

    Evaluates retrieval and generation quality.
    """

    def __init__(self, llm_judge: LLMProtocol | None = None):
        self._llm = llm_judge

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGScore:
        """
        Evaluate RAG output.

        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Expected answer (optional)

        Returns:
            RAG evaluation scores
        """
        score = RAGScore()

        if self._llm:
            # Use LLM for evaluation
            score.context_relevance = await self._evaluate_context_relevance(question, contexts)
            score.answer_relevance = await self._evaluate_answer_relevance(question, answer)
            score.answer_faithfulness = await self._evaluate_faithfulness(answer, contexts)
        else:
            # Use heuristic evaluation
            score.context_relevance = self._heuristic_context_relevance(question, contexts)
            score.answer_relevance = self._heuristic_answer_relevance(question, answer)
            score.answer_faithfulness = self._heuristic_faithfulness(answer, contexts)

        # Calculate precision/recall if ground truth available
        if ground_truth:
            score.context_precision = self._calculate_precision(ground_truth, contexts)
            score.context_recall = self._calculate_recall(ground_truth, contexts)
            score.answer_completeness = self._calculate_completeness(answer, ground_truth)

        # Calculate overall score
        scores = [
            score.context_relevance,
            score.answer_relevance,
            score.answer_faithfulness,
        ]
        if ground_truth:
            scores.extend(
                [
                    score.context_precision,
                    score.context_recall,
                    score.answer_completeness,
                ]
            )

        score.overall_score = sum(scores) / len(scores) if scores else 0.0

        return score

    async def _evaluate_context_relevance(
        self,
        question: str,
        contexts: list[str],
    ) -> float:
        """Evaluate context relevance using LLM."""
        if not contexts:
            return 0.0

        context_text = "\n\n".join(f"Context {i+1}: {c}" for i, c in enumerate(contexts))

        prompt = f"""Rate how relevant the retrieved contexts are to the question.

Question: {question}

{context_text}

Rate from 0 to 10 where:
0 = Completely irrelevant
5 = Somewhat relevant
10 = Highly relevant

Output only a number between 0 and 10:"""

        response = await self._llm.complete(prompt)
        return self._parse_score(response) / 10

    async def _evaluate_answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> float:
        """Evaluate answer relevance using LLM."""
        prompt = f"""Rate how well the answer addresses the question.

Question: {question}

Answer: {answer}

Rate from 0 to 10 where:
0 = Does not address the question
5 = Partially addresses the question
10 = Fully addresses the question

Output only a number between 0 and 10:"""

        response = await self._llm.complete(prompt)
        return self._parse_score(response) / 10

    async def _evaluate_faithfulness(
        self,
        answer: str,
        contexts: list[str],
    ) -> float:
        """Evaluate answer faithfulness to context."""
        if not contexts:
            return 0.0

        context_text = "\n\n".join(contexts)

        prompt = f"""Rate how faithful the answer is to the given context.
Check if all claims in the answer can be verified from the context.

Context:
{context_text}

Answer: {answer}

Rate from 0 to 10 where:
0 = Contains many claims not in context (hallucination)
5 = Some claims verifiable, some not
10 = All claims can be verified from context

Output only a number between 0 and 10:"""

        response = await self._llm.complete(prompt)
        return self._parse_score(response) / 10

    def _heuristic_context_relevance(
        self,
        question: str,
        contexts: list[str],
    ) -> float:
        """Heuristic context relevance."""
        if not contexts:
            return 0.0

        question_terms = set(question.lower().split())

        scores = []
        for context in contexts:
            context_terms = set(context.lower().split())
            overlap = len(question_terms & context_terms)
            score = overlap / len(question_terms) if question_terms else 0
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _heuristic_answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> float:
        """Heuristic answer relevance."""
        question_terms = set(question.lower().split())
        answer_terms = set(answer.lower().split())

        overlap = len(question_terms & answer_terms)
        relevance = overlap / len(question_terms) if question_terms else 0

        # Boost if answer is substantial
        if len(answer) > 50:
            relevance = min(relevance + 0.2, 1.0)

        return relevance

    def _heuristic_faithfulness(
        self,
        answer: str,
        contexts: list[str],
    ) -> float:
        """Heuristic faithfulness check."""
        if not contexts:
            return 0.0

        context_text = " ".join(contexts).lower()
        answer_sentences = re.split(r"[.!?]", answer)

        verified = 0
        total = 0

        for sentence in answer_sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue

            total += 1
            sentence_terms = set(sentence.lower().split())
            context_terms = set(context_text.split())

            overlap = len(sentence_terms & context_terms)
            if overlap / len(sentence_terms) > 0.5:
                verified += 1

        return verified / total if total > 0 else 0.5

    def _calculate_precision(self, ground_truth: str, contexts: list[str]) -> float:
        """Calculate retrieval precision."""
        if not contexts:
            return 0.0

        truth_terms = set(ground_truth.lower().split())

        relevant = 0
        for context in contexts:
            context_terms = set(context.lower().split())
            if len(truth_terms & context_terms) / len(truth_terms) > 0.3:
                relevant += 1

        return relevant / len(contexts)

    def _calculate_recall(self, ground_truth: str, contexts: list[str]) -> float:
        """Calculate retrieval recall."""
        if not contexts:
            return 0.0

        truth_terms = set(ground_truth.lower().split())
        context_text = " ".join(contexts).lower()
        context_terms = set(context_text.split())

        covered = len(truth_terms & context_terms)
        return covered / len(truth_terms) if truth_terms else 0.0

    def _calculate_completeness(self, answer: str, ground_truth: str) -> float:
        """Calculate answer completeness."""
        truth_terms = set(ground_truth.lower().split())
        answer_terms = set(answer.lower().split())

        covered = len(truth_terms & answer_terms)
        return covered / len(truth_terms) if truth_terms else 0.0

    def _parse_score(self, response: str) -> float:
        """Parse numeric score from LLM response."""
        numbers = re.findall(r"\d+\.?\d*", response)
        if numbers:
            score = float(numbers[0])
            return min(max(score, 0), 10)
        return 5.0
