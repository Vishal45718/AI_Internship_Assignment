"""
app/evaluation/benchmark.py — Retrieval quality and hallucination evaluation.

This module provides programmatic evaluation tools for:
1. Retrieval accuracy — does the right chunk come back for known queries?
2. Hallucination detection — does the system refuse to answer out-of-scope questions?
3. Confidence threshold tuning — what threshold minimizes false positives?

Run: python -m app.evaluation.benchmark
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Test Case Definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalTestCase:
    """A test case for retrieval accuracy evaluation."""
    query: str
    expected_source: str           # Expected source filename (partial match OK)
    should_retrieve: bool = True   # False = this query SHOULD trigger fallback


@dataclass
class HallucinationTestCase:
    """A test case to verify the system refuses to hallucinate."""
    query: str
    description: str               # What makes this a hallucination-risk query


# ─────────────────────────────────────────────────────────────────────────────
# Default test suite (works with the provided sample documents)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RETRIEVAL_TESTS: list[RetrievalTestCase] = [
    RetrievalTestCase(
        query="What is the return policy for electronics?",
        expected_source="sample_sop.txt",
    ),
    RetrievalTestCase(
        query="How do I reset my password?",
        expected_source="sample_faq.txt",
    ),
    RetrievalTestCase(
        query="What are the system requirements?",
        expected_source="sample_guide.md",
    ),
    RetrievalTestCase(
        query="xkc29fj quantum teleportation in 1842",   # garbage query
        expected_source="",
        should_retrieve=False,
    ),
    RetrievalTestCase(
        query="What is the weather in Tokyo today?",     # out-of-scope
        expected_source="",
        should_retrieve=False,
    ),
]

DEFAULT_HALLUCINATION_TESTS: list[HallucinationTestCase] = [
    HallucinationTestCase(
        query="What was the stock price of NVIDIA on March 15, 2023?",
        description="Specific financial data not in documents",
    ),
    HallucinationTestCase(
        query="Who is the CEO of the company?",
        description="Person-specific fact unlikely in sample docs",
    ),
    HallucinationTestCase(
        query="What happened in the French Revolution?",
        description="Historical fact unrelated to documents",
    ),
    HallucinationTestCase(
        query="Generate a poem about machine learning",
        description="Creative task — should be declined or redirected",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    test_name: str
    passed: int = 0
    failed: int = 0
    total: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"{self.test_name}: {self.passed}/{self.total} passed "
            f"({self.pass_rate:.0%})"
        )


class EvaluationBenchmark:
    """
    Runs the evaluation suite against the live RAG system.

    Usage:
        benchmark = EvaluationBenchmark(agent, vector_store)
        results = benchmark.run_all()
        benchmark.print_report(results)
    """

    def __init__(self, agent, vector_store) -> None:
        self._agent = agent
        self._store = vector_store

    def run_retrieval_tests(
        self,
        test_cases: list[RetrievalTestCase] | None = None,
    ) -> BenchmarkResult:
        """
        Test that retrieval returns the expected source documents.
        """
        cases = test_cases or DEFAULT_RETRIEVAL_TESTS
        result = BenchmarkResult(test_name="Retrieval Accuracy")

        for case in cases:
            start = time.perf_counter()
            response = self._agent.query(case.query)
            latency_ms = (time.perf_counter() - start) * 1000

            if case.should_retrieve:
                # Passed if the expected source appears in citations
                sources = [s.source_file for s in response.sources]
                passed = any(case.expected_source in s for s in sources)
                reason = f"Expected '{case.expected_source}' in {sources}"
            else:
                # Passed if the system triggered a fallback (not a confident answer)
                from app.models.responses import ResponseStatus
                passed = response.status in (
                    ResponseStatus.FALLBACK, ResponseStatus.OUT_OF_SCOPE
                )
                reason = f"Status was {response.status.value}"

            detail = {
                "query": case.query[:80],
                "passed": passed,
                "status": response.status.value,
                "latency_ms": round(latency_ms, 1),
                "reason": reason,
            }
            result.details.append(detail)
            result.total += 1
            if passed:
                result.passed += 1
            else:
                result.failed += 1
                logger.warning("FAILED retrieval test: %s | %s", case.query[:60], reason)

        return result

    def run_hallucination_tests(
        self,
        test_cases: list[HallucinationTestCase] | None = None,
    ) -> BenchmarkResult:
        """
        Verify the system refuses to hallucinate for out-of-scope queries.
        A "pass" means the response is a fallback/out-of-scope, NOT a confident answer.
        """
        from app.models.responses import ResponseStatus

        cases = test_cases or DEFAULT_HALLUCINATION_TESTS
        result = BenchmarkResult(test_name="Hallucination Prevention")

        for case in cases:
            response = self._agent.query(case.query)
            passed = response.status in (
                ResponseStatus.FALLBACK, ResponseStatus.OUT_OF_SCOPE
            )
            detail = {
                "query": case.query[:80],
                "description": case.description,
                "passed": passed,
                "status": response.status.value,
                "confidence": response.confidence,
                "answer_preview": (response.answer or "")[:100],
            }
            result.details.append(detail)
            result.total += 1
            if passed:
                result.passed += 1
            else:
                result.failed += 1
                logger.warning(
                    "HALLUCINATION RISK: '%s' returned status=%s with confidence=%.3f",
                    case.query[:60],
                    response.status.value,
                    response.confidence or 0.0,
                )

        return result

    def run_all(self) -> list[BenchmarkResult]:
        """Run the complete evaluation suite."""
        logger.info("Starting evaluation benchmark…")
        results = [
            self.run_retrieval_tests(),
            self.run_hallucination_tests(),
        ]
        return results

    @staticmethod
    def print_report(results: list[BenchmarkResult], output_path: str | None = None) -> None:
        """Print a human-readable benchmark report."""
        lines = ["", "=" * 60, "  AGENTIC RAG — EVALUATION REPORT", "=" * 60]

        for result in results:
            lines.append(f"\n{result.summary()}")
            lines.append("-" * 50)
            for detail in result.details:
                status_icon = "✓" if detail.get("passed") else "✗"
                query = detail.get("query", "")[:60]
                lines.append(f"  {status_icon} {query}")
                if not detail.get("passed"):
                    lines.append(f"    → {detail.get('reason', detail.get('status', ''))}")

        overall_passed = sum(r.passed for r in results)
        overall_total = sum(r.total for r in results)
        overall_rate = overall_passed / overall_total if overall_total > 0 else 0
        lines.append(f"\nOVERALL: {overall_passed}/{overall_total} ({overall_rate:.0%})")
        lines.append("=" * 60)

        report = "\n".join(lines)
        print(report)

        if output_path:
            Path(output_path).write_text(report)
            logger.info("Report saved to: %s", output_path)

        # Also save JSON for programmatic consumption
        if output_path:
            json_path = output_path.replace(".txt", ".json")
            data = [
                {
                    "test_name": r.test_name,
                    "pass_rate": r.pass_rate,
                    "passed": r.passed,
                    "total": r.total,
                    "details": r.details,
                }
                for r in results
            ]
            Path(json_path).write_text(json.dumps(data, indent=2))
