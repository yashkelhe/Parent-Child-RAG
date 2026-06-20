"""
ingestion/child_chunker.py

Splits each ParentChunk into smaller "child" chunks (~250-400 tokens, with
overlap) that will be embedded and stored in Pinecone. Each child resolves
its own page_number by mapping its character offset (within the parent,
projected back into the full document) against the page-offset table built
during parent chunking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.parent_chunker import PageOffset, ParentChunk
from utils.helper import count_tokens, get_logger, new_id

logger = get_logger(__name__)

_CHARS_PER_TOKEN = 4


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    chunk_index: int
    page_number: int
    text: str
    document_name: str  # carried along for convenience / Pinecone metadata
    token_count: int

    def to_storage_dict(self) -> dict:
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "text": self.text,
        }


class ChildChunker:
    def __init__(
        self,
        min_tokens: int = 250,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
    ) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_parent(self, parent: ParentChunk, page_offsets: List[PageOffset]) -> List[ChildChunk]:
        if not parent.text.strip():
            return []

        target_chars = int(((self.min_tokens + self.max_tokens) / 2) * _CHARS_PER_TOKEN)
        overlap_chars = int(self.overlap_tokens * _CHARS_PER_TOKEN)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        pieces = splitter.split_text(parent.text)

        children: List[ChildChunk] = []
        cursor = 0  # offset within parent.text, used to project to absolute offsets
        for idx, piece in enumerate(pieces):
            if not piece.strip():
                continue
            local_idx = parent.text.find(piece, max(0, cursor - overlap_chars))
            if local_idx == -1:
                local_idx = cursor
            absolute_offset = parent.start_char + local_idx
            page_number = self._page_for_offset(absolute_offset, page_offsets)

            children.append(
                ChildChunk(
                    child_id=new_id("child_"),
                    parent_id=parent.parent_id,
                    chunk_index=idx,
                    page_number=page_number,
                    text=piece.strip(),
                    document_name=parent.document_name,
                    token_count=count_tokens(piece),
                )
            )
            cursor = local_idx + len(piece)

        logger.debug(
            "Parent '%s' split into %d child chunks.", parent.parent_id, len(children)
        )
        return children

    def chunk_parents(self, parents: List[ParentChunk], page_offsets: List[PageOffset]) -> List[ChildChunk]:
        all_children: List[ChildChunk] = []
        for parent in parents:
            all_children.extend(self.chunk_parent(parent, page_offsets))
        logger.info("Built %d child chunks across %d parents.", len(all_children), len(parents))
        return all_children

    @staticmethod
    def _page_for_offset(offset: int, page_offsets: List[PageOffset]) -> int:
        for po in page_offsets:
            if po.start <= offset < po.end:
                return po.page_number
        if page_offsets:
            # offset fell in a separator between pages - snap to nearest page
            nearest = min(page_offsets, key=lambda po: min(abs(po.start - offset), abs(po.end - offset)))
            return nearest.page_number
        return 1
