#!/usr/bin/env python3
"""
run.py — CLI entry point for the RAG system.

Commands:
    python run.py ingest data/raw/              # Ingest documents from a directory
    python run.py ingest report.pdf faq.txt     # Ingest specific files
    python run.py chat                          # Start interactive Q&A session
    python run.py chat "What is X?"             # One-shot query
    python run.py stats                         # Show index statistics
    python run.py reset                         # Reset the vector index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logging import setup_logging


def cmd_ingest(args, pipeline) -> int:
    """Ingest documents into the vector store."""
    if not args.paths:
        from src.config import get_settings
        settings = get_settings()
        default_dir = settings.data_dir
        if default_dir.exists() and any(default_dir.iterdir()):
            print(f"No paths given. Using default: {default_dir}")
            args.paths = [str(default_dir)]
        else:
            print("No paths specified and data/raw/ is empty.")
            print("  Usage: python run.py ingest <file_or_directory>")
            return 1

    total_chunks = 0
    total_files = 0

    for path_str in args.paths:
        p = Path(path_str)
        if not p.exists():
            print(f"  ⚠️  Not found: {path_str}")
            continue

        print(f"\n{'📁' if p.is_dir() else '📄'} Ingesting: {p}")
        result = pipeline.ingest(p)

        if result["status"] == "success":
            chunks = result.get("chunks", 0)
            total_chunks += chunks
            total_files += result.get("documents", 1)
            print(f"  ✓ {chunks} chunks created")
        elif result["status"] == "warning":
            print(f"  ⚠️  {result['message']}")
        else:
            print(f"  ✗ Error: {result.get('message', 'Unknown error')}")

    # Summary
    print(f"\n{'─' * 50}")
    print(f"✓ Ingestion complete: {total_files} file(s), {total_chunks} chunks")
    _print_stats(pipeline)
    return 0


def cmd_chat(args, pipeline) -> int:
    """Interactive Q&A or one-shot query."""
    # One-shot mode
    if args.query:
        query = " ".join(args.query)
        result = pipeline.query(query)
        _print_answer(result, show_debug=args.debug)
        return 0 if result["status"] == "success" else 1

    # Interactive REPL mode
    print("\n🤖  RAG System — Interactive Q&A")
    print("   Ask questions about your ingested documents.")
    print("   Commands: /quit, /stats, /help\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Special commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            elif cmd == "/stats":
                _print_stats(pipeline)
            elif cmd == "/help":
                print("\nCommands:")
                print("  /quit   — Exit")
                print("  /stats  — Show index statistics")
                print("  /help   — Show this help\n")
            else:
                print(f"Unknown command: {user_input}")
            continue

        # Regular query
        result = pipeline.query(user_input)
        print("\nAssistant:")
        _print_answer(result, show_debug=args.debug)

    return 0


def cmd_stats(args, pipeline) -> int:
    """Show index statistics."""
    _print_stats(pipeline)
    return 0


def cmd_reset(args, pipeline) -> int:
    """Reset the vector index."""
    confirm = input("⚠️  This will delete ALL indexed chunks. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        pipeline.reset_index()
        print("✓ Vector index has been reset.")
    else:
        print("Reset cancelled.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_answer(result: dict, show_debug: bool = False) -> None:
    """Pretty-print a query result."""
    status = result.get("status", "unknown")
    icons = {"success": "✓", "no_relevant_context": "⚠️", "empty_index": "📭", "error": "✗"}
    icon = icons.get(status, "?")

    print(f"\n{icon} [{status.upper()}]")
    print("─" * 60)
    print(result["answer"])

    # Source citations
    sources = result.get("sources", [])
    if sources:
        print("\n📚 Sources:")
        for i, src in enumerate(sources, 1):
            page = f" (page {src['page']})" if src.get("page") else ""
            print(f"  [{i}] {src['file']}{page}  — score: {src['score']:.3f}")
            if show_debug:
                print(f"      Preview: {src['preview'][:120]}")

    # Debug info
    if show_debug and "confidence" in result:
        print(f"\n🔍 Confidence: {result['confidence']:.3f}")

    print()


def _print_stats(pipeline) -> None:
    """Print index statistics."""
    stats = pipeline.get_stats()
    print(f"\n📊 Index Status:")
    print(f"  Total chunks    : {stats['total_chunks']}")
    print(f"  Total documents : {stats['total_documents']}")
    if stats["documents"]:
        print("  Indexed files:")
        for doc in stats["documents"]:
            print(f"    • {doc['source_file']:<40} {doc['chunk_count']:>4} chunks")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RAG System — Document Q&A with hallucination prevention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents into the vector store")
    p_ingest.add_argument("paths", nargs="*", help="File or directory paths to ingest")

    # chat
    p_chat = subparsers.add_parser("chat", help="Interactive Q&A or one-shot query")
    p_chat.add_argument("query", nargs="*", help="Question (omit for interactive mode)")
    p_chat.add_argument("--debug", "-d", action="store_true", help="Show debug info")

    # stats
    subparsers.add_parser("stats", help="Show index statistics")

    # reset
    subparsers.add_parser("reset", help="Reset the vector index")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Setup logging
    setup_logging("DEBUG" if args.verbose else "WARNING")

    # Initialize pipeline (lazy — only loads models when needed)
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline()

    # Dispatch command
    commands = {
        "ingest": cmd_ingest,
        "chat": cmd_chat,
        "stats": cmd_stats,
        "reset": cmd_reset,
    }
    handler = commands.get(args.command)
    if handler:
        return handler(args, pipeline)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
