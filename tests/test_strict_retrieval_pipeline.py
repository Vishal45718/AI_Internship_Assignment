import unittest

from src.config import get_settings
from src.models import RetrievedChunk
from src.retrieval.retriever import SemanticRetriever


class FakeReranker:
    def score(self, query: str, passages):
        lowered_query = query.lower()
        scores = []
        for passage in passages:
            text = passage.lower()
            score = 0.05
            if "seakr" in lowered_query and "seakr" in text:
                score += 0.45
            if "dragin" in lowered_query and "dragin" in text:
                score += 0.45
            if "real" in lowered_query and "real" in text:
                score += 0.45
            if "query-level feedback" in lowered_query and "feedback" in text:
                score += 0.4
            if any(word in text for word in ("threshold", "uncertainty", "reasoning", "token importance", "vocabulary mismatch", "ambiguity", "hallucination", "noise")):
                score += 0.25
            scores.append(min(1.0, score))
        return scores


class StrictMockStore:
    def __init__(self) -> None:
        self.chunks = [
            RetrievedChunk(
                chunk_id="seakr_1",
                content="SeaKR computes self-aware uncertainty from internal states of LLM generation.",
                source_file="RAG.pdf",
                source_type="pdf",
                page_number=7,
                chunk_index=10,
                similarity_score=0.86,
            ),
            RetrievedChunk(
                chunk_id="seakr_2",
                content="When uncertainty crosses a threshold, retrieval is triggered for grounded generation.",
                source_file="RAG.pdf",
                source_type="pdf",
                page_number=7,
                chunk_index=11,
                similarity_score=0.84,
            ),
            RetrievedChunk(
                chunk_id="dragin_1",
                content="DRAGIN reformulates retrieval queries using token probabilities and attention weights.",
                source_file="RAG.pdf",
                source_type="pdf",
                page_number=4,
                chunk_index=5,
                similarity_score=0.88,
            ),
            RetrievedChunk(
                chunk_id="dragin_2",
                content="The reformulation uses reasoning traces while excluding uncertain tokens.",
                source_file="RAG.pdf",
                source_type="pdf",
                page_number=4,
                chunk_index=6,
                similarity_score=0.82,
            ),
            RetrievedChunk(
                chunk_id="real_1",
                content="ReAL optimizes token importance via adaptive learning for query expansion.",
                source_file="RAG.pdf",
                source_type="pdf",
                page_number=5,
                chunk_index=7,
                similarity_score=0.9,
            ),
            RetrievedChunk(
                chunk_id="feedback_1",
                content="Query-level feedback suffers from vocabulary mismatch and ambiguity.",
                source_file="RAG.pdf",
                source_type="pdf",
                page_number=9,
                chunk_index=20,
                similarity_score=0.85,
            ),
            RetrievedChunk(
                chunk_id="feedback_2",
                content="It can propagate hallucination and noise into subsequent retrieval.",
                source_file="RAG.pdf",
                source_type="pdf",
                page_number=9,
                chunk_index=21,
                similarity_score=0.83,
            ),
            RetrievedChunk(
                chunk_id="noise_1",
                content="Unrelated appendix text with weak relevance.",
                source_file="RAG.pdf",
                source_type="pdf",
                page_number=1,
                chunk_index=0,
                similarity_score=0.3,
            ),
        ]

    def count(self) -> int:
        return len(self.chunks)

    def query(self, query_text: str, top_k: int | None = None, where=None):
        return sorted(self.chunks, key=lambda c: c.similarity_score, reverse=True)[: top_k or len(self.chunks)]

    def keyword_search(self, query_text: str, top_k: int | None = None, where=None):
        terms = query_text.lower().split()
        matched = [c for c in self.chunks if any(t in c.content.lower() for t in terms)]
        return sorted(matched, key=lambda c: c.similarity_score, reverse=True)[: top_k or len(matched)]

    def get_neighboring_chunks(self, chunk_ids, window_before=1, window_after=1):
        index_map = {c.chunk_id: c for c in self.chunks}
        neighbors = {}
        for cid in chunk_ids:
            seed = index_map.get(cid)
            if seed is None:
                continue
            start = seed.chunk_index - window_before
            end = seed.chunk_index + window_after
            for chunk in self.chunks:
                if chunk.source_file == seed.source_file and start <= chunk.chunk_index <= end:
                    neighbors[chunk.chunk_id] = chunk
        return list(neighbors.values())

    def get_chunks_by_section(self, source_file, section_title):
        return []


class StrictRetrievalPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = get_settings()
        self.originals = {
            "hybrid": self.settings.hybrid_search_enabled,
            "retrieval_top_k": self.settings.retrieval_top_k,
            "rerank_top_k": self.settings.rerank_top_k,
            "final_context_chunks": self.settings.final_context_chunks,
            "max_context_tokens": self.settings.max_context_tokens,
            "retrieval_score_threshold": self.settings.retrieval_score_threshold,
            "rerank_score_threshold": self.settings.rerank_score_threshold,
            "parent_context_enabled": self.settings.parent_context_enabled,
            "parent_window_before": self.settings.parent_window_before,
            "parent_window_after": self.settings.parent_window_after,
        }
        self.settings.hybrid_search_enabled = True
        self.settings.retrieval_top_k = 20
        self.settings.rerank_top_k = 5
        self.settings.final_context_chunks = 5
        self.settings.max_context_tokens = 1500
        self.settings.retrieval_score_threshold = 0.55
        self.settings.rerank_score_threshold = 0.20
        self.settings.parent_context_enabled = True
        self.settings.parent_window_before = 1
        self.settings.parent_window_after = 1

        self.retriever = SemanticRetriever(StrictMockStore())
        self.retriever._reranker = FakeReranker()

    def tearDown(self) -> None:
        for key, value in self.originals.items():
            setattr(self.settings, key, value)

    def test_seakr_query(self):
        result = self.retriever.retrieve("Explain the consistency used in SeaKR.", expand_context=True)
        text = " ".join(c.content for c in result.chunks).lower()
        self.assertIn("self-aware uncertainty", text)
        self.assertIn("internal states", text)
        self.assertIn("threshold", text)

    def test_dragin_query(self):
        result = self.retriever.retrieve("How does DRAGIN reformulate retrieval queries?", expand_context=True)
        text = " ".join(c.content for c in result.chunks).lower()
        self.assertIn("token probabilities", text)
        self.assertIn("attention weights", text)
        self.assertIn("reasoning", text)
        self.assertIn("excluding uncertain tokens", text)

    def test_real_query(self):
        result = self.retriever.retrieve("What does ReAL optimize?", expand_context=True)
        text = " ".join(c.content for c in result.chunks).lower()
        self.assertIn("token importance", text)
        self.assertIn("adaptive learning", text)
        self.assertIn("query expansion", text)

    def test_query_feedback_limitations(self):
        result = self.retriever.retrieve("What are the limitations of query-level feedback?", expand_context=True)
        text = " ".join(c.content for c in result.chunks).lower()
        self.assertIn("vocabulary mismatch", text)
        self.assertIn("ambiguity", text)
        self.assertIn("hallucination", text)
        self.assertIn("noise", text)


if __name__ == "__main__":
    unittest.main()
