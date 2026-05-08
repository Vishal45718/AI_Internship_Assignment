#!/usr/bin/env python3
"""
scripts/evaluate.py — Run the evaluation benchmark suite.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --output reports/benchmark_results.txt
    python scripts/evaluate.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation benchmark.")
    parser.add_argument("--output", "-o", default=None, help="Path to save report.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")

    from app.core.dependencies import get_rag_agent, get_vector_store
    from app.evaluation.benchmark import EvaluationBenchmark

    agent = get_rag_agent()
    store = get_vector_store()

    if store.count() == 0:
        print("⚠️  Vector index is empty. Please ingest documents before running evaluation.")
        print("   python scripts/ingest.py data/raw/")
        return 1

    benchmark = EvaluationBenchmark(agent=agent, vector_store=store)
    results = benchmark.run_all()
    benchmark.print_report(results, output_path=args.output)

    # Exit code: 0 if all tests pass, 1 if any fail
    all_passed = all(r.passed == r.total for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
