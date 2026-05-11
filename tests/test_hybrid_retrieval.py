import unittest

from src.config import get_settings
from src.models import RetrievedChunk
from src.retrieval.retriever import SemanticRetriever


class MockStore:
    def count(self) -> int:
        return 2

    def query(self, query_text: str, top_k: int | None = None, where=None) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="semantic_1",
                content="This chunk discusses natural language understanding and general AI behavior.",
                source_file="doc.txt",
                source_type="txt",
                page_number=None,
                chunk_index=0,
                similarity_score=0.92,
            ),
            RetrievedChunk(
                chunk_id="exact_real",
                content="ReAL is a reinforcement learning algorithm used for robust adaptation.",
                source_file="doc.txt",
                source_type="txt",
                page_number=None,
                chunk_index=1,
                similarity_score=0.08,
            ),
        ]

    def keyword_search(self, query_text: str, top_k: int | None = None, where=None) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="exact_real",
                content="ReAL is a reinforcement learning algorithm used for robust adaptation.",
                source_file="doc.txt",
                source_type="txt",
                page_number=None,
                chunk_index=1,
                similarity_score=0.95,
            )
        ]


class HybridRetrievalTest(unittest.TestCase):
    def test_exact_acronym_with_hybrid_search(self):
        settings = get_settings()
        original_value = settings.hybrid_search_enabled
        settings.hybrid_search_enabled = True
        try:
            retriever = SemanticRetriever(MockStore())
            result = retriever.retrieve("What is ReAL?", top_k=5)

            self.assertTrue(result.passed_threshold)
            self.assertGreater(len(result.chunks), 0)
            self.assertEqual(result.chunks[0].chunk_id, "exact_real")
            self.assertTrue(any("ReAL" in chunk.content for chunk in result.chunks))
        finally:
            settings.hybrid_search_enabled = original_value


if __name__ == "__main__":
    unittest.main()
