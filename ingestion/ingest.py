"""
ingestion/ingest.py

Orchestrates the full ingestion pipeline for one or more PDFs:

PDF -> PDFLoader -> DocumentNormalizer -> ParentChunker -> save parents.json
    -> ChildChunker -> GeminiEmbedder -> PineconeDB.upsert
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from config import settings
from ingestion.child_chunker import ChildChunk, ChildChunker
from ingestion.document_normalizer import DocumentNormalizer
from ingestion.embedder import GeminiEmbedder
from ingestion.parent_chunker import ParentChunk, ParentChunker
from ingestion.pdf_loader import PDFLoadError, PDFLoader
from utils.helper import get_logger, load_json, save_json
from vectorstore.pinecone_db import PineconeDB

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        embedder: GeminiEmbedder | None = None,
        vector_store: PineconeDB | None = None,
        parents_path: Path | None = None,
    ) -> None:
        self.embedder = embedder or GeminiEmbedder()
        self.vector_store = vector_store or PineconeDB()
        self.vector_store.connect()
        self.parents_path = parents_path or settings.storage.parents_json_path
        self.normalizer = DocumentNormalizer()

    def ingest_file(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        document_name = file_path.name
        logger.info("=== Ingesting '%s' ===", document_name)

        try:
            pages = PDFLoader(file_path).load()
        except PDFLoadError as exc:
            logger.error("Aborting ingestion of '%s': %s", document_name, exc)
            return

        normalized_pages = self.normalizer.normalize(pages)

        parent_chunker = ParentChunker(
            document_name=document_name,
            min_tokens=settings.chunking.parent_min_tokens,
            max_tokens=settings.chunking.parent_max_tokens,
        )
        parents, page_offsets = parent_chunker.chunk(normalized_pages)
        if not parents:
            logger.warning("No parent chunks produced for '%s'; skipping.", document_name)
            return

        self._persist_parents(parents)

        child_chunker = ChildChunker(
            min_tokens=settings.chunking.child_min_tokens,
            max_tokens=settings.chunking.child_max_tokens,
            overlap_tokens=settings.chunking.child_overlap_tokens,
        )
        children = child_chunker.chunk_parents(parents, page_offsets)
        if not children:
            logger.warning("No child chunks produced for '%s'; nothing to embed.", document_name)
            return

        self._embed_and_upsert(children)
        logger.info("=== Finished ingesting '%s' ===", document_name)

    def ingest_directory(self, directory: str | Path) -> None:
        directory = Path(directory)
        pdf_files = sorted(directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in '%s'.", directory)
            return
        for pdf_file in pdf_files:
            self.ingest_file(pdf_file)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _persist_parents(self, new_parents: List[ParentChunk]) -> None:
        existing = load_json(self.parents_path, default=[])
        existing.extend(p.to_storage_dict() for p in new_parents)
        save_json(self.parents_path, existing)
        logger.info(
            "Persisted %d new parent chunks to '%s' (total: %d).",
            len(new_parents),
            self.parents_path,
            len(existing),
        )

    def _embed_and_upsert(self, children: List[ChildChunk]) -> None:
        texts = [c.text for c in children]
        embeddings = self.embedder.embed_batch(texts, task_type="retrieval_document")

        ids = [c.child_id for c in children]
        metadatas = [
            {
                "child_id": c.child_id,
                "parent_id": c.parent_id,
                "page_number": c.page_number,
                "document_name": c.document_name,
                "chunk_index": c.chunk_index,
                "child_text": c.text,
            }
            for c in children
        ]

        self.vector_store.upsert_children(ids=ids, embeddings=embeddings, metadatas=metadatas)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDFs into the Parent-Child RAG system.")
    parser.add_argument("path", help="Path to a PDF file or a directory of PDFs.")
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    target = Path(args.path)
    if target.is_dir():
        pipeline.ingest_directory(target)
    else:
        pipeline.ingest_file(target)


if __name__ == "__main__":
    main()
