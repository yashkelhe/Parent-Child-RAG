"""
ingestion/embedder.py

Thin wrapper around the Gemini embeddings API (google-genai SDK). Generates
embeddings ONLY for child chunks (parents are never embedded, per the
architecture).
"""

from __future__ import annotations

from typing import List

from google import genai
from google.genai import types

from config import settings
from utils.helper import chunked, get_logger

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails irrecoverably."""


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.api_key = api_key or settings.gemini.api_key
        self.model = model or settings.gemini.embed_model
        self.batch_size = batch_size

        if not self.api_key:
            raise EmbeddingError("GOOGLE_API_KEY is not set. Cannot initialize embedder.")

        self._client = genai.Client(api_key=self.api_key)

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Embed a single string. Returns an empty list for blank input."""
        if not text or not text.strip():
            return []
        try:
            response = self._client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return list(response.embeddings[0].values)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc

    def embed_query(self, query: str) -> List[float]:
        """Embed a user query (uses the RETRIEVAL_QUERY task type for asymmetric search)."""
        return self.embed_text(query, task_type="RETRIEVAL_QUERY")

    def embed_batch(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """
        Embed many texts, batching requests and skipping/blanking out any
        individual failures rather than aborting the whole ingestion run.
        """
        embeddings: List[List[float]] = []
        batches = chunked(texts, self.batch_size)

        for batch_index, batch in enumerate(batches):
            for text in batch:
                if not text or not text.strip():
                    logger.warning("Skipping embedding for empty text (batch %d).", batch_index)
                    embeddings.append([])
                    continue
                try:
                    vector = self.embed_text(text, task_type=task_type)
                    embeddings.append(vector)
                except EmbeddingError as exc:
                    logger.error("Embedding failed for one chunk in batch %d: %s", batch_index, exc)
                    embeddings.append([])
            logger.info("Embedded batch %d/%d (%d items).", batch_index + 1, len(batches), len(batch))

        return embeddings
