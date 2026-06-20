"""
app.py

Top-level CLI for the Parent-Child RAG system.

Usage:
    python app.py ingest data/                # ingest all PDFs in a folder
    python app.py ingest data/handbook.pdf     # ingest a single PDF
    python app.py query "What is the leave policy?"
    python app.py chat                         # interactive Q&A loop
"""

from __future__ import annotations

import argparse
import sys

from config import settings
from ingestion.ingest import IngestionPipeline
from retrieval.retriever import RAGAnswer, Retriever
from utils.helper import get_logger

logger = get_logger(__name__, level=settings.log_level)


def run_ingest(path: str) -> None:
    pipeline = IngestionPipeline()
    from pathlib import Path

    target = Path(path)
    if target.is_dir():
        pipeline.ingest_directory(target)
    else:
        pipeline.ingest_file(target)


def run_query(question: str) -> None:
    retriever = Retriever()
    result: RAGAnswer = retriever.answer(question)
    _print_answer(result)


def run_chat() -> None:
    retriever = Retriever()
    print("Parent-Child RAG — interactive mode. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        try:
            result = retriever.answer(question)
            _print_answer(result)
        except Exception as exc:
            logger.error("Failed to answer question: %s", exc)
            print(f"Sorry, something went wrong: {exc}\n")


def _print_answer(result: RAGAnswer) -> None:
    print("\n--- Answer ---")
    print(result.answer)
    if result.used_parents:
        print("\n--- Sources ---")
        for parent in result.used_parents:
            print(f"- {parent.document_name} | {parent.title} | p.{parent.page_start}-{parent.page_end}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parent-Child RAG system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF file or directory of PDFs.")
    ingest_parser.add_argument("path", help="Path to a PDF file or directory.")

    query_parser = subparsers.add_parser("query", help="Ask a single question.")
    query_parser.add_argument("question", help="The question to ask.")

    subparsers.add_parser("chat", help="Start an interactive Q&A session.")

    args = parser.parse_args()

    try:
        if args.command == "ingest":
            run_ingest(args.path)
        elif args.command == "query":
            run_query(args.question)
        elif args.command == "chat":
            run_chat()
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
