import unittest

from src.config import get_settings
from src.models import RetrievedChunk
from src.retrieval.retriever import SemanticRetriever


class MockStore:
    def count(self) -> int:
        return 2

    def query(self, query_text: str, top_k: int | None = None, where=None) -> list[RetrievedChunk]:
        generic_intro = RetrievedChunk(
            chunk_id="intro_1",
            content="Introduction to the paper: general overview of retrieval and model design.",
            source_file="doc.txt",
            source_type="txt",
            page_number=1,
            chunk_index=0,
            similarity_score=0.92,
        )
        seakr_chunk = RetrievedChunk(
            chunk_id="seakr_method",
            content="SeaKR introduces a stability-aware ranking process for query reformulation.",
            source_file="doc.txt",
            source_type="txt",
            page_number=3,
            chunk_index=4,
            similarity_score=0.45,
        )
        flare_chunk = RetrievedChunk(
            chunk_id="flare_method",
            content="FLARE presents a query-aware fusion architecture and token interaction analysis.",
            source_file="doc.txt",
            source_type="txt",
            page_number=4,
            chunk_index=2,
            similarity_score=0.43,
        )
        dragin_chunk = RetrievedChunk(
            chunk_id="dragin_method",
            content="DRAGIN reformulates retrieval queries with token probabilities and attention weights.",
            source_file="doc.txt",
            source_type="txt",
            page_number=5,
            chunk_index=1,
            similarity_score=0.40,
        )
        dragin_details = RetrievedChunk(
            chunk_id="dragin_detail",
            content="Token probability estimation and uncertain token handling are core parts of DRAGIN.",
            source_file="doc.txt",
            source_type="txt",
            page_number=5,
            chunk_index=2,
            similarity_score=0.38,
        )
        real_chunk = RetrievedChunk(
            chunk_id="exact_real",
            content="ReAL is a reinforcement learning algorithm used for robust adaptation.",
            source_file="doc.txt",
            source_type="txt",
            page_number=2,
            chunk_index=1,
            similarity_score=0.08,
        )
        real_token_chunk = RetrievedChunk(
            chunk_id="real_token",
            content="Token importance ranking and query expansion are key components of ReAL.",
            source_file="doc.txt",
            source_type="txt",
            page_number=2,
            chunk_index=2,
            similarity_score=0.10,
        )

        if "ReAL" in query_text:
            return [generic_intro, real_chunk, real_token_chunk]
        if "DRAGIN" in query_text and "Compare" not in query_text:
            return [generic_intro, dragin_chunk, dragin_details]
        if "SeaKR" in query_text or "FLARE" in query_text or "DRAGIN" in query_text:
            return [generic_intro, seakr_chunk, flare_chunk, dragin_chunk, dragin_details]

        return [generic_intro, real_chunk]

    def keyword_search(self, query_text: str, top_k: int | None = None, where=None) -> list[RetrievedChunk]:
        results: list[RetrievedChunk] = []
        if "ReAL" in query_text:
            results.append(
                RetrievedChunk(
                    chunk_id="exact_real",
                    content="ReAL is a reinforcement learning algorithm used for robust adaptation.",
                    source_file="doc.txt",
                    source_type="txt",
                    page_number=2,
                    chunk_index=1,
                    similarity_score=0.95,
                )
            )
            results.append(
                RetrievedChunk(
                    chunk_id="real_token",
                    content="Token importance ranking and query expansion are key components of ReAL.",
                    source_file="doc.txt",
                    source_type="txt",
                    page_number=2,
                    chunk_index=2,
                    similarity_score=0.72,
                )
            )
        if "DRAGIN" in query_text:
            results.append(
                RetrievedChunk(
                    chunk_id="dragin_method",
                    content="DRAGIN reformulates retrieval queries with token probabilities and attention weights.",
                    source_file="doc.txt",
                    source_type="txt",
                    page_number=5,
                    chunk_index=1,
                    similarity_score=0.93,
                )
            )
            results.append(
                RetrievedChunk(
                    chunk_id="dragin_detail",
                    content="Token probability estimation and uncertain token handling are core parts of DRAGIN.",
                    source_file="doc.txt",
                    source_type="txt",
                    page_number=5,
                    chunk_index=2,
                    similarity_score=0.88,
                )
            )
        if "SeaKR" in query_text:
            results.append(
                RetrievedChunk(
                    chunk_id="seakr_method",
                    content="SeaKR introduces a stability-aware ranking process for query reformulation.",
                    source_file="doc.txt",
                    source_type="txt",
                    page_number=3,
                    chunk_index=4,
                    similarity_score=0.90,
                )
            )
        if "FLARE" in query_text:
            results.append(
                RetrievedChunk(
                    chunk_id="flare_method",
                    content="FLARE presents a query-aware fusion architecture and token interaction analysis.",
                    source_file="doc.txt",
                    source_type="txt",
                    page_number=4,
                    chunk_index=2,
                    similarity_score=0.91,
                )
            )
        return results

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
        if "seakr_method" in chunk_ids and window_before > 0:
            neighbors.append(
                RetrievedChunk(
                    chunk_id="seakr_before",
                    content="Background on SeaKR and section-level motivation.",
                    source_file="doc.txt",
                    source_type="txt",
                    page_number=3,
                    chunk_index=3,
                    similarity_score=0.80,
                )
            )
        if "dragin_method" in chunk_ids and window_before > 0:
            neighbors.append(
                RetrievedChunk(
                    chunk_id="dragin_before",
                    content="Adjacent context for DRAGIN method descriptions.",
                    source_file="doc.txt",
                    source_type="txt",
                    page_number=5,
                    chunk_index=0,
                    similarity_score=0.80,
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
        original_threshold = settings.retrieval_score_threshold
        settings.hybrid_search_enabled = True
        settings.parent_context_enabled = False
        settings.parent_expand_full_section = False
        settings.retrieval_score_threshold = 0.55
        try:
            retriever = SemanticRetriever(MockStore())
            result = retriever.retrieve("What does ReAL optimize?", top_k=5, expand_context=False)

            self.assertTrue(result.passed_threshold)
            self.assertGreater(len(result.chunks), 0)
            chunk_ids = [c.chunk_id for c in result.chunks]
            self.assertIn("exact_real", chunk_ids, "Expected exact_real chunk in hybrid results")
            self.assertIn("real_token", chunk_ids, "Expected ReAL token importance chunk in hybrid results")
            self.assertTrue(any("ReAL" in chunk.content for chunk in result.chunks))
        finally:
            settings.hybrid_search_enabled = original_value
            settings.parent_context_enabled = original_parent
            settings.parent_expand_full_section = original_section
            settings.retrieval_score_threshold = original_threshold

    def test_draging_mechanism_retrieval(self):
        settings = get_settings()
        original_value = settings.hybrid_search_enabled
        original_parent = settings.parent_context_enabled
        original_threshold = settings.retrieval_score_threshold
        settings.hybrid_search_enabled = True
        settings.parent_context_enabled = False
        settings.retrieval_score_threshold = 0.40
        try:
            retriever = SemanticRetriever(MockStore())
            result = retriever.retrieve(
                "How does DRAGIN reformulate retrieval queries?",
                top_k=10,
                expand_context=False,
            )

            self.assertTrue(result.passed_threshold)
            chunk_ids = [c.chunk_id for c in result.chunks]
            self.assertIn("dragin_method", chunk_ids)
            self.assertIn("dragin_detail", chunk_ids)
            self.assertFalse(all(chunk.page_number == 1 for chunk in result.chunks), "Expected fewer page 1 intro chunks")
        finally:
            settings.hybrid_search_enabled = original_value
            settings.parent_context_enabled = original_parent
            settings.retrieval_score_threshold = original_threshold

    def test_compare_multiple_methods_with_diversity(self):
        settings = get_settings()
        original_value = settings.hybrid_search_enabled
        original_parent = settings.parent_context_enabled
        original_threshold = settings.retrieval_score_threshold
        settings.hybrid_search_enabled = True
        settings.parent_context_enabled = False
        settings.retrieval_score_threshold = 0.40
        try:
            retriever = SemanticRetriever(MockStore())
            result = retriever.retrieve(
                "Compare SeaKR, FLARE, and DRAGIN.",
                top_k=10,
                expand_context=False,
            )

            self.assertTrue(result.passed_threshold)
            chunk_ids = [c.chunk_id for c in result.chunks]
            self.assertIn("seakr_method", chunk_ids)
            self.assertIn("flare_method", chunk_ids)
            self.assertIn("dragin_method", chunk_ids)
            pages = [c.page_number for c in result.chunks]
            self.assertLessEqual(len([p for p in pages if p == 1]), 2, "Expected at most two page 1 chunks")
            self.assertGreaterEqual(len(set(pages)), 2, "Expected multiple pages in top results")
        finally:
            settings.hybrid_search_enabled = original_value
            settings.parent_context_enabled = original_parent
            settings.retrieval_score_threshold = original_threshold


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
        original_threshold = settings.retrieval_score_threshold
        settings.parent_context_enabled = False
        settings.parent_expand_full_section = False
        settings.retrieval_score_threshold = 0.55

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
            settings.retrieval_score_threshold = original_threshold


if __name__ == "__main__":
    unittest.main()
