from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker with preferred model and fallback."""

    def __init__(self) -> None:
        self._model = None
        self.model_name = ""
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            logger.warning("CrossEncoder unavailable: %s", exc)
            return

        candidates = [
            "BAAI/bge-reranker-base",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ]
        for name in candidates:
            try:
                self._model = CrossEncoder(name, max_length=512, local_files_only=True)
                self.model_name = name
                logger.info("Loaded reranker model: %s", name)
                return
            except Exception as exc:
                logger.warning("Failed loading reranker %s locally: %s", name, exc)

    def score(self, query: str, passages: Iterable[str]) -> list[float]:
        texts = list(passages)
        if not texts:
            return []
        if self._model is None:
            return self._fallback_scores(query, texts)

        pairs = [(query, text) for text in texts]
        try:
            raw_scores = self._model.predict(pairs)
        except Exception as exc:
            logger.warning("Reranker predict failed, using fallback: %s", exc)
            return self._fallback_scores(query, texts)

        normalized: list[float] = []
        for value in raw_scores:
            score = float(value)
            if score < 0:
                # map roughly from [-10,10] into [0,1]
                score = 1 / (1 + pow(2.718281828, -score))
            normalized.append(max(0.0, min(1.0, score)))
        return normalized

    def _fallback_scores(self, query: str, passages: list[str]) -> list[float]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        scores: list[float] = []
        for passage in passages:
            lowered = passage.lower()
            overlap = sum(1 for term in query_terms if term in lowered)
            coverage = overlap / max(len(query_terms), 1)
            explanatory_bonus = 0.15 if any(
                token in lowered for token in ("because", "therefore", "threshold", "method", "optimiz", "consisten")
            ) else 0.0
            scores.append(max(0.0, min(1.0, coverage + explanatory_bonus)))
        return scores
