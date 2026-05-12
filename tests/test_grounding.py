"""Unit tests for strict grounding helpers."""

import unittest

from src.llm.grounding import (
    assess_pre_generation_support,
    protected_acronyms_in_query,
    validate_post_generation,
)
from src.models import RetrievedChunk


class GroundingTests(unittest.TestCase):
    def test_protected_acronyms_detected(self) -> None:
        self.assertEqual(protected_acronyms_in_query("What is SeaKR?"), ["SeaKR"])
        self.assertIn("ReAL", protected_acronyms_in_query("Compare ReAL and DRAGIN."))

    def test_pre_generation_blocks_missing_entity(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk_id="a",
                content="Generic discussion of neural networks and optimization.",
                source_file="x.pdf",
                source_type="pdf",
                similarity_score=0.9,
                rerank_score=0.85,
            )
        ]
        ok, reasons, _ = assess_pre_generation_support(
            "What is SeaKR?",
            chunks,
            top_score=0.85,
            rerank_score_threshold=0.2,
        )
        self.assertFalse(ok)
        self.assertTrue(any("required_entity_missing_in_evidence" in r for r in reasons))

    def test_post_generation_accepts_disclaimer(self) -> None:
        ctx = "SeaKR computes uncertainty from internal states."
        ans = "The retrieved documents do not contain enough information."
        v = validate_post_generation(ans, "Explain SeaKR", ctx, 0.8)
        self.assertTrue(v["ok"])
        self.assertTrue(v["is_insufficient_disclaimer"])

    def test_post_generation_flags_unsupported_terms(self) -> None:
        ctx = "SeaKR computes uncertainty from internal states."
        ans = (
            "SeaKR is kernel regularization applied to Bayesian neural networks in PyTorch "
            "with stochastic depth."
        )
        v = validate_post_generation(ans, "What is SeaKR?", ctx, 0.8)
        self.assertFalse(v["ok"])
        self.assertTrue(v["regenerate"])
        self.assertTrue(any("kernel regularization" in t for t in v.get("hallucination_triggers", [])))


if __name__ == "__main__":
    unittest.main()
