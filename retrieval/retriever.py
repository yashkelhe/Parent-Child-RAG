"""
retrieval/retriever.py

Orchestrates the retrieval half of the pipeline:
embed query -> search Pinecone (child chunks) -> collect unique parent_ids
in relevance order -> load parent chunks from parents.json -> build prompt
-> call Gemini -> return answer with source metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from config import settings
from ingestion.embedder import GeminiEmbedder
from ingestion.parent_chunker import ParentChunk
from llm.gemini import GeminiLLM
from retrieval.prompt_builder import PromptBuilder
from utils.helper import get_logger, load_json
from vectorstore.pinecone_db import ChildSearchResult, PineconeDB

logger = get_logger(__name__)


class RetrievalError(Exception):
    """Raised when retrieval cannot proceed (e.g. parents.json missing)."""


@dataclass
class RAGAnswer:
    answer: str
    question: str
    used_parents: List[ParentChunk]
    top_children: List[ChildSearchResult]


class ParentStore:
    """Read-only accessor over parents.json, keyed by parent_id."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or settings.storage.parents_json_path
        self._by_id: Dict[str, ParentChunk] = {}
        self._load()

    def _load(self) -> None:
        raw = load_json(self.path, default=[])
        for item in raw:
            try:
                parent = ParentChunk(
                    parent_id=item["parent_id"],
                    title=item["title"],
                    document_name=item["document_name"],
                    page_start=item["page_start"],
                    page_end=item["page_end"],
                    text=item["text"],
                    token_count=item["token_count"],
                )
                self._by_id[parent.parent_id] = parent
            except KeyError as exc:
                logger.warning("Skipping malformed parent record (missing field %s): %s", exc, item)
        logger.info("Loaded %d parent chunks from '%s'.", len(self._by_id), self.path)

    def get(self, parent_id: str) -> Optional[ParentChunk]:
        return self._by_id.get(parent_id)

    def reload(self) -> None:
        self._by_id.clear()
        self._load()


class Retriever:
    def __init__(
        self,
        embedder: Optional[GeminiEmbedder] = None,
        vector_store: Optional[PineconeDB] = None,
        llm: Optional[GeminiLLM] = None,
        parent_store: Optional[ParentStore] = None,
        top_k_children: Optional[int] = None,
        max_parent_contexts: Optional[int] = None,
    ) -> None:
        self.embedder = embedder or GeminiEmbedder()
        self.vector_store = vector_store or PineconeDB()
        self.vector_store.connect()
        self.llm = llm or GeminiLLM()
        self.parent_store = parent_store or ParentStore()
        self.top_k_children = top_k_children or settings.retrieval.top_k_children
        self.max_parent_contexts = max_parent_contexts or settings.retrieval.max_parent_contexts

    def answer(self, question: str) -> RAGAnswer:
        if not question or not question.strip():
            raise RetrievalError("Question must be a non-empty string.")

        query_embedding = self.embedder.embed_query(question)
        if not query_embedding:
            raise RetrievalError("Failed to embed the query - cannot search.")

        top_children = self.vector_store.query(query_embedding, top_k=self.top_k_children)
        if not top_children:
            logger.warning("No child chunks matched the query.")
            prompt = PromptBuilder.build(question, [])
            answer_text = self.llm.generate(prompt)
            return RAGAnswer(answer=answer_text, question=question, used_parents=[], top_children=[])

        ordered_parent_ids = self._unique_parent_ids_by_relevance(top_children)
        parents = self._load_parents(ordered_parent_ids[: self.max_parent_contexts])

        if not parents:
            logger.warning("Matched child chunks but found no corresponding parent records.")

        prompt = PromptBuilder.build(question, parents)
        answer_text = self.llm.generate(prompt)

        return RAGAnswer(
            answer=answer_text,
            question=question,
            used_parents=parents,
            top_children=top_children,
        )

    @staticmethod
    def _unique_parent_ids_by_relevance(children: List[ChildSearchResult]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for child in children:
            if child.parent_id and child.parent_id not in seen:
                seen.add(child.parent_id)
                ordered.append(child.parent_id)
        return ordered

    def _load_parents(self, parent_ids: List[str]) -> List[ParentChunk]:
        parents = []
        for pid in parent_ids:
            parent = self.parent_store.get(pid)
            if parent is None:
                logger.warning("Parent id '%s' referenced by a child but missing from parents.json.", pid)
                continue
            parents.append(parent)
        return parents
