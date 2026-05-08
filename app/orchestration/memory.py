"""
app/orchestration/memory.py — Conversation memory with rolling window.

Maintains per-session conversation history so follow-up questions
(e.g., "What about their return policy?") work correctly after
previous context has been established.

Design: A simple rolling window (last N turns) stored in memory.
For production: replace the dict store with Redis for multi-instance support.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque

from configs.settings import get_settings

logger = logging.getLogger(__name__)


class ConversationTurn:
    """A single turn in a conversation."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role        # "user" or "assistant"
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """
    Rolling conversation window for a single session.

    Keeps at most `max_turns` individual messages (not turn-pairs).
    When the window is full, the oldest message is dropped.
    """

    def __init__(self, session_id: str, max_turns: int | None = None) -> None:
        settings = get_settings()
        self._session_id = session_id
        self._max_messages = (max_turns or settings.max_conversation_turns) * 2
        self._history: Deque[ConversationTurn] = deque(maxlen=self._max_messages)

    def add_user(self, content: str) -> None:
        self._history.append(ConversationTurn("user", content))
        logger.debug("[%s] Added user turn.", self._session_id)

    def add_assistant(self, content: str) -> None:
        self._history.append(ConversationTurn("assistant", content))
        logger.debug("[%s] Added assistant turn.", self._session_id)

    def get_history(self) -> list[dict[str, str]]:
        """Return the current window as a list of {"role": ..., "content": ...} dicts."""
        # Exclude the most recent user message (it's passed separately as the query)
        turns = list(self._history)
        if turns and turns[-1].role == "user":
            turns = turns[:-1]
        return [t.to_dict() for t in turns]

    def clear(self) -> None:
        """Reset this session's memory."""
        self._history.clear()
        logger.info("[%s] Conversation memory cleared.", self._session_id)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return len(self._history) // 2


class MemoryStore:
    """
    In-process store for multiple conversation sessions.

    Usage:
        store = MemoryStore()
        session = store.get_or_create("session-123")
        session.add_user("Hello")
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationMemory] = {}

    def get_or_create(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(session_id)
            logger.info("New conversation session created: %s", session_id)
        return self._sessions[session_id]

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Conversation session deleted: %s", session_id)
            return True
        return False

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    @property
    def session_count(self) -> int:
        return len(self._sessions)
