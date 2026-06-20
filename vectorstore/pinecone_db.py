"""
vectorstore/pinecone_db.py

Wraps the Pinecone client: index creation/connection, upserting child
embeddings with metadata, and querying for the most similar children.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

from config import settings
from utils.helper import get_logger

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised on unrecoverable Pinecone errors."""


@dataclass
class ChildSearchResult:
    child_id: str
    parent_id: str
    score: float
    page_number: int
    document_name: str
    chunk_index: int
    child_text: str


class PineconeDB:
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.pinecone.api_key
        self.index_name = index_name or settings.pinecone.index_name
        self.dimension = dimension or settings.gemini.embedding_dim
        self.cloud = cloud or settings.pinecone.cloud
        self.region = region or settings.pinecone.region
        self.metric = metric or settings.pinecone.metric

        if not self.api_key:
            raise VectorStoreError("PINECONE_API_KEY is not set. Cannot initialize vector store.")

        self._client = Pinecone(api_key=self.api_key)
        self._index = None

    def connect(self) -> None:
        """Create the index if it doesn't exist yet, then attach to it."""
        try:
            existing = {idx["name"] for idx in self._client.list_indexes()}
        except Exception as exc:
            raise VectorStoreError(f"Could not list Pinecone indexes: {exc}") from exc

        if self.index_name not in existing:
            logger.info(
                "Index '%s' not found. Creating serverless index (dim=%d, metric=%s, %s/%s)...",
                self.index_name,
                self.dimension,
                self.metric,
                self.cloud,
                self.region,
            )
            try:
                self._client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
            except Exception as exc:
                raise VectorStoreError(f"Failed to create Pinecone index '{self.index_name}': {exc}") from exc
        else:
            logger.info("Connecting to existing Pinecone index '%s'.", self.index_name)

        self._index = self._client.Index(self.index_name)

    def _require_index(self):
        if self._index is None:
            raise VectorStoreError("PineconeDB.connect() must be called before using the index.")
        return self._index

    def upsert_children(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Upsert (id, embedding, metadata) triples. Skips entries with empty embeddings."""
        index = self._require_index()

        vectors = []
        skipped = 0
        for vec_id, embedding, metadata in zip(ids, embeddings, metadatas):
            if not embedding:
                skipped += 1
                continue
            vectors.append({"id": vec_id, "values": embedding, "metadata": metadata})

        if skipped:
            logger.warning("Skipped %d vectors with empty/failed embeddings during upsert.", skipped)

        if not vectors:
            logger.warning("No valid vectors to upsert.")
            return 0

        upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            try:
                index.upsert(vectors=batch)
                upserted += len(batch)
            except Exception as exc:
                raise VectorStoreError(f"Failed to upsert batch starting at index {i}: {exc}") from exc

        logger.info("Upserted %d vectors into index '%s'.", upserted, self.index_name)
        return upserted

    def query(self, embedding: List[float], top_k: int = 8) -> List[ChildSearchResult]:
        if not embedding:
            raise VectorStoreError("Cannot query Pinecone with an empty embedding.")

        index = self._require_index()
        try:
            response = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        except Exception as exc:
            raise VectorStoreError(f"Pinecone query failed: {exc}") from exc

        results: List[ChildSearchResult] = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {}) or {}
            results.append(
                ChildSearchResult(
                    child_id=match.get("id", ""),
                    parent_id=metadata.get("parent_id", ""),
                    score=match.get("score", 0.0),
                    page_number=metadata.get("page_number", -1),
                    document_name=metadata.get("document_name", ""),
                    chunk_index=metadata.get("chunk_index", -1),
                    child_text=metadata.get("child_text", ""),
                )
            )
        return results
