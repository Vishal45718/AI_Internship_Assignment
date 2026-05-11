import unittest
from src.config import get_settings
from src.models import RetrievedChunk
from src.retrieval.retriever import SemanticRetriever

class MockStore:
    def __init__(self):
        # We will mock a sequence of 5 chunks
        # Chunk index 0: background
        # Chunk index 1: SeaKR setup
        # Chunk index 2: SeaKR computations (this matches query)
        # Chunk index 3: threshold and consistency explanation (neighboring)
        # Chunk index 4: ReAL token importance (matches another query)
        # Chunk index 5: ReAL neighboring explanation text
        self.chunks = [
            RetrievedChunk(
                chunk_id="chunk_0",
                content="Background on LLM hallucination.",
                source_file="doc.txt",
                source_type="txt",
                chunk_index=0,
                similarity_score=0.1
            ),
            RetrievedChunk(
                chunk_id="chunk_1",
                content="SeaKR is a method for RAG.",
                source_file="doc.txt",
                source_type="txt",
                chunk_index=1,
                similarity_score=0.3
            ),
            RetrievedChunk(
                chunk_id="chunk_2",
                content="SeaKR computes self-aware uncertainty for retrieval.",
                source_file="doc.txt",
                source_type="txt",
                chunk_index=2,
                similarity_score=0.9
            ),
            RetrievedChunk(
                chunk_id="chunk_3",
                content="The threshold determines consistency used in SeaKR.",
                source_file="doc.txt",
                source_type="txt",
                chunk_index=3,
                similarity_score=0.4
            ),
            RetrievedChunk(
                chunk_id="chunk_4",
                content="ReAL optimizes token importance for query expansion.",
                source_file="doc.txt",
                source_type="txt",
                chunk_index=4,
                similarity_score=0.88
            ),
            RetrievedChunk(
                chunk_id="chunk_5",
                content="Neighboring explanation text for ReAL.",
                source_file="doc.txt",
                source_type="txt",
                chunk_index=5,
                similarity_score=0.2
            )
        ]

    def count(self) -> int:
        return len(self.chunks)

    def query(self, query_text: str, top_k: int | None = None, where=None) -> list[RetrievedChunk]:
        if "SeaKR" in query_text:
            return [self.chunks[2]]
        elif "ReAL" in query_text:
            return [self.chunks[4]]
        return []

    def get_chunks_by_indices(self, source_file: str, indices: list[int]) -> list[RetrievedChunk]:
        return [c for c in self.chunks if c.source_file == source_file and c.chunk_index in indices]

    def get_neighboring_chunks(
        self,
        chunk_ids: list[str],
        window_before: int = 1,
        window_after: int = 1,
    ) -> list[RetrievedChunk]:
        by_id = {c.chunk_id: c for c in self.chunks}
        neighbors: dict[str, RetrievedChunk] = {}
        for chunk_id in chunk_ids:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            start = max(0, chunk.chunk_index - window_before)
            end = chunk.chunk_index + window_after
            for candidate in self.chunks:
                if (
                    candidate.source_file == chunk.source_file
                    and start <= candidate.chunk_index <= end
                ):
                    neighbors[candidate.chunk_id] = candidate
        return list(neighbors.values())

    def get_chunks_by_section(self, source_file: str, section_title: str) -> list[RetrievedChunk]:
        return []


class ParentContextRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.settings = get_settings()
        self.original_enabled = self.settings.parent_context_enabled
        self.original_before = self.settings.parent_window_before
        self.original_after = self.settings.parent_window_after
        self.original_section = self.settings.parent_expand_full_section
        self.original_threshold = self.settings.retrieval_score_threshold
        self.original_hybrid = self.settings.hybrid_search_enabled
        
        self.settings.parent_context_enabled = True
        self.settings.parent_window_before = 1
        self.settings.parent_window_after = 1
        self.settings.parent_expand_full_section = False
        self.settings.retrieval_score_threshold = 0.55
        self.settings.hybrid_search_enabled = False

    def tearDown(self):
        self.settings.parent_context_enabled = self.original_enabled
        self.settings.parent_window_before = self.original_before
        self.settings.parent_window_after = self.original_after
        self.settings.parent_expand_full_section = self.original_section
        self.settings.retrieval_score_threshold = self.original_threshold
        self.settings.hybrid_search_enabled = self.original_hybrid

    def test_seakr_consistency(self):
        retriever = SemanticRetriever(MockStore())
        result = retriever.retrieve("Explain the consistency used in SeaKR.", expand_context=True)
        
        self.assertTrue(result.passed_threshold)
        # Should include chunk 1, 2, 3
        contents = " ".join([c.content for c in result.chunks])
        
        self.assertIn("SeaKR", contents)
        self.assertIn("uncertainty", contents)
        self.assertIn("threshold", contents)
        self.assertIn("consistency", contents)
        
        self.assertGreater(result.expanded_context_token_count, 0)

    def test_real_optimization(self):
        retriever = SemanticRetriever(MockStore())
        result = retriever.retrieve("What does ReAL optimize?", expand_context=True)
        
        self.assertTrue(result.passed_threshold)
        # Should include chunk 3, 4, 5
        contents = " ".join([c.content for c in result.chunks])
        
        self.assertIn("token importance", contents)
        self.assertIn("query expansion", contents)
        self.assertIn("Neighboring explanation text", contents)

if __name__ == "__main__":
    unittest.main()
