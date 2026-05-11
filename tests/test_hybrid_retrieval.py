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

    def get_chunks_by_indices(self, source_file: str, indices: list[int]) -> list[RetrievedChunk]:
        # Return chunks from query/keyword_search that match the requested index
        all_chunks = self.query("", 0) + self.keyword_search("", 0)
        return [c for c in all_chunks if c.source_file == source_file and c.chunk_index in indices]

    def get_neighboring_chunks(
        self, chunk_ids: list[str], window_before: int = 1, window_after: int = 1
    ) -> list[RetrievedChunk]:
        neighbors: list[RetrievedChunk] = []
        if "semantic_1" in chunk_ids and window_before > 0:
            neighbors.append(
                RetrievedChunk(
                    chunk_id="semantic_0",
                    content="Section: Advanced Learning Algorithms. This section covers...",
                    source_file="doc.txt",
                    source_type="txt",
                    page_number=1,
                    chunk_index=-1,
                    similarity_score=0.85,
                )
            )
        if "semantic_1" in chunk_ids and window_after > 0:
            neighbors.append(
                RetrievedChunk(
                    chunk_id="semantic_2",
                    content="The algorithm achieves state-of-the-art performance on multiple benchmarks...",
                    source_file="doc.txt",
                    source_type="txt",
                    page_number=1,
                    chunk_index=2,
                    similarity_score=0.88,
                )
            )
        if "exact_real" in chunk_ids and window_before > 0:
            neighbors.append(
                RetrievedChunk(
                    chunk_id="exact_real_before",
                    content="Token importance ranking and query expansion are key components of ReAL.",
                    source_file="doc.txt",
                    source_type="txt",
                    page_number=1,
                    chunk_index=0,
                    similarity_score=0.90,
                )
            )
        return neighbors

    def get_chunks_by_section(self, source_file: str, section_title: str) -> list[RetrievedChunk]:
        return []

class HybridRetrievalTest(unittest.TestCase):
    def test_exact_acronym_with_hybrid_search(self):
        settings = get_settings()
        original_value = settings.hybrid_search_enabled
        original_parent = settings.parent_context_enabled
        original_section = settings.parent_expand_full_section
        original_threshold = settings.similarity_threshold
        settings.hybrid_search_enabled = True
        settings.parent_context_enabled = False
        settings.parent_expand_full_section = False
        settings.similarity_threshold = 0.70
        try:
            retriever = SemanticRetriever(MockStore())
            result = retriever.retrieve("What is ReAL?", top_k=5, expand_context=False)

            self.assertTrue(result.passed_threshold)
            self.assertGreater(len(result.chunks), 0)
            chunk_ids = [c.chunk_id for c in result.chunks]
            self.assertIn("exact_real", chunk_ids, "Expected exact_real chunk in hybrid results")
            self.assertTrue(any("ReAL" in chunk.content for chunk in result.chunks))
        finally:
            settings.hybrid_search_enabled = original_value
            settings.parent_context_enabled = original_parent
            settings.parent_expand_full_section = original_section
            settings.similarity_threshold = original_threshold


class ParentDocumentRetrievalTest(unittest.TestCase):
    def test_context_expansion_with_parent_retrieval(self):
        settings = get_settings()
        original_enabled = settings.parent_context_enabled
        original_before = settings.parent_window_before
        original_after = settings.parent_window_after
        original_section = settings.parent_expand_full_section

        settings.parent_context_enabled = True
        settings.parent_window_before = 1
        settings.parent_window_after = 1
        settings.parent_expand_full_section = False

        try:
            retriever = SemanticRetriever(MockStore())
            result = retriever.retrieve("Explain the consistency used in SeaKR.", expand_context=True)

            self.assertTrue(result.passed_threshold)
            self.assertGreater(len(result.chunks), 0)
            chunk_ids = [c.chunk_id for c in result.chunks]
            self.assertTrue(
                len(chunk_ids) > 1,
                f"Expected expanded context with multiple chunks, got {len(chunk_ids)} chunk(s)",
            )
        finally:
            settings.parent_context_enabled = original_enabled
            settings.parent_window_before = original_before
            settings.parent_window_after = original_after
            settings.parent_expand_full_section = original_section

    def test_context_expansion_disabled(self):
        settings = get_settings()
        original_enabled = settings.parent_context_enabled
        original_section = settings.parent_expand_full_section
        original_threshold = settings.similarity_threshold
        settings.parent_context_enabled = False
        settings.parent_expand_full_section = False
        settings.similarity_threshold = 0.70

        try:
            retriever = SemanticRetriever(MockStore())
            result = retriever.retrieve(
                "Explain the consistency used in SeaKR.",
                expand_context=False,
            )

            self.assertTrue(result.passed_threshold)
            self.assertEqual(len(result.chunks), 1, "Expected single chunk when expansion is disabled")
        finally:
            settings.parent_context_enabled = original_enabled
            settings.parent_expand_full_section = original_section
            settings.similarity_threshold = original_threshold


if __name__ == "__main__":
    unittest.main()
