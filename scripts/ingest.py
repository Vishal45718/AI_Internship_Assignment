#!/usr/bin/env python3
"""
scripts/ingest.py — CLI for document ingestion.

Usage:
    python scripts/ingest.py data/raw/                          # ingest a directory
    python scripts/ingest.py document.pdf report.txt            # ingest specific files
    python scripts/ingest.py data/raw/ --recursive              # recursive directory scan
    python scripts/ingest.py --stats                            # show current index stats
    python scripts/ingest.py --reset                            # reset the vector index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.logging import setup_logging
from configs.settings import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Agentic RAG vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="File paths or directories to ingest.",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively scan directories.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current index statistics and exit.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="⚠️  Reset the entire vector index (deletes all chunks).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging("DEBUG" if args.verbose else settings.log_level)

    # Import after logging is configured
    from app.core.dependencies import get_embedder, get_vector_store, get_ingestion_pipeline

    store = get_vector_store()
    pipeline = get_ingestion_pipeline()

    # ── --stats ───────────────────────────────────────────────────────────────
    if args.stats:
        _print_stats(store)
        return 0

    # ── --reset ───────────────────────────────────────────────────────────────
    if args.reset:
        confirm = input("⚠️  This will delete ALL indexed chunks. Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            store.reset()
            print("✓ Vector index has been reset.")
        else:
            print("Reset cancelled.")
        return 0

    # ── Ingest paths ──────────────────────────────────────────────────────────
    if not args.paths:
        # Default to data/raw if no paths provided
        default_dir = settings.data_dir
        if default_dir.exists() and any(default_dir.iterdir()):
            print(f"No paths specified. Using default directory: {default_dir}")
            args.paths = [str(default_dir)]
        else:
            print("No paths specified and data/raw/ is empty.")
            print(f"  Usage: python scripts/ingest.py <file_or_directory>")
            return 1

    total_chunks = 0
    total_files = 0
    failed = []

    for path_str in args.paths:
        path = Path(path_str)
        if path.is_dir():
            print(f"\n📁 Ingesting directory: {path}")
            result = pipeline.ingest_directory(path, recursive=args.recursive)
            total_chunks += result.total_chunks
            total_files += result.success_count
            failed.extend(result.failed_files)
        elif path.is_file():
            print(f"\n📄 Ingesting file: {path.name}")
            result = pipeline.ingest_files([str(path)])
            total_chunks += result.total_chunks
            total_files += result.success_count
            failed.extend(result.failed_files)
        else:
            print(f"  ⚠️  Path not found: {path}")
            failed.append(str(path))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print(f"✓ Ingestion complete")
    print(f"  Files processed : {total_files}")
    print(f"  Chunks created  : {total_chunks}")
    print(f"  Failed          : {len(failed)}")
    if failed:
        print("  Failed files:")
        for f in failed[:10]:
            print(f"    - {f}")

    _print_stats(store)
    return 0 if not failed else 1


def _print_stats(store) -> None:
    """Print current index statistics."""
    sources = store.list_sources()
    total_chunks = store.count()
    print(f"\n📊 Index Status:")
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Total documents : {len(sources)}")
    if sources:
        print("  Indexed files:")
        for s in sources:
            print(f"    • {s['source_file']:<40} {s['chunk_count']:>4} chunks")


if __name__ == "__main__":
    sys.exit(main())
