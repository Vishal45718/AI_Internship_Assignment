#!/usr/bin/env python3
"""
scripts/query.py — Interactive CLI for querying the RAG system.

Usage:
    python scripts/query.py                           # interactive REPL
    python scripts/query.py "What is the return policy?"   # one-shot query
    python scripts/query.py --session my-session      # with conversation memory
    python scripts/query.py --debug                   # show retrieval details
    python scripts/query.py --no-rerank               # disable reranking
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.logging import setup_logging
from configs.settings import get_settings


def _print_response(response, show_debug: bool = False) -> None:
    """Pretty-print a QueryResponse to the terminal."""
    from app.models.responses import ResponseStatus

    # Status indicator
    icons = {
        ResponseStatus.SUCCESS: "✓",
        ResponseStatus.FALLBACK: "⚠️ ",
        ResponseStatus.OUT_OF_SCOPE: "🚫",
        ResponseStatus.ERROR: "✗",
    }
    icon = icons.get(response.status, "?")

    print(f"\n{icon} [{response.status.value.upper()}]")
    print("─" * 60)
    print(response.answer)

    # Source citations
    if response.sources:
        print("\n📚 Sources:")
        for i, src in enumerate(response.sources, 1):
            page = f" (page {src.page_number})" if src.page_number else ""
            print(f"  [{i}] {src.source_file}{page}  — score: {src.relevance_score:.3f}")
            if show_debug:
                print(f"      Preview: {src.preview[:120]}")

    # Debug info
    if show_debug:
        print(f"\n🔍 Debug:")
        print(f"  Intent            : {response.intent or 'n/a'}")
        print(f"  Confidence        : {response.confidence:.3f}" if response.confidence else "  Confidence        : n/a")
        print(f"  Retrieval Strategy: {response.retrieval_strategy or 'n/a'}")
        print(f"  Latency           : {response.latency_ms:.0f}ms" if response.latency_ms else "  Latency: n/a")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query the Agentic RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to ask (omit for interactive REPL mode).",
    )
    parser.add_argument(
        "--session", "-s",
        default=None,
        help="Session ID for conversation memory (auto-generated if not provided).",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="Disable conversation memory (stateless queries).",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Show retrieval debug information.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: from config).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    settings = get_settings()
    # In non-debug CLI mode, suppress INFO logs so only the response is shown
    log_level = "DEBUG" if args.verbose else "WARNING"
    setup_logging(log_level)

    from app.core.dependencies import get_rag_agent

    agent = get_rag_agent()

    # Determine session ID
    if args.no_session:
        session_id = None
    else:
        session_id = args.session or str(uuid.uuid4())[:8]
        if not args.query:
            print(f"Session ID: {session_id}  (use --session {session_id} to resume)")

    # ── One-shot mode ─────────────────────────────────────────────────────────
    if args.query:
        response = agent.query(
            query=args.query,
            session_id=session_id,
            top_k=args.top_k,
            enable_reranking=not args.no_rerank,
        )
        _print_response(response, show_debug=args.debug)
        return 0 if response.status.value != "error" else 1

    # ── Interactive REPL mode ─────────────────────────────────────────────────
    print("\n🤖  Agentic RAG — Interactive Query Mode")
    print("   Type your question and press Enter.")
    print("   Commands: /quit, /clear (clear memory), /stats, /help")
    print(f"   Session: {session_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Special commands ──────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            elif cmd == "/clear":
                from app.core.dependencies import get_rag_agent
                # Reset memory for this session
                agent._memory.delete(session_id)
                session_id = str(uuid.uuid4())[:8]
                print(f"✓ Memory cleared. New session: {session_id}")
            elif cmd == "/stats":
                from app.core.dependencies import get_vector_store
                store = get_vector_store()
                sources = store.list_sources()
                print(f"\n📊 Index: {store.count()} chunks | {len(sources)} documents")
                for s in sources:
                    print(f"  • {s['source_file']} ({s['chunk_count']} chunks)")
                print()
            elif cmd == "/help":
                print("\nCommands:")
                print("  /quit, /exit  — Exit the REPL")
                print("  /clear        — Clear conversation memory")
                print("  /stats        — Show index statistics")
                print("  /help         — Show this help message\n")
            else:
                print(f"Unknown command: {user_input}")
            continue

        # ── Regular query ─────────────────────────────────────────────────────
        response = agent.query(
            query=user_input,
            session_id=session_id,
            top_k=args.top_k,
            enable_reranking=not args.no_rerank,
        )
        print("\nAssistant:", end="")
        _print_response(response, show_debug=args.debug)

    return 0


if __name__ == "__main__":
    sys.exit(main())
