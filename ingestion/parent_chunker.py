"""
ingestion/parent_chunker.py

Builds large "parent" chunks (~1000-2000 tokens) from normalized page text.

Strategy:
1. Concatenate all pages into one document string, while recording the
   character-offset range each page occupies (needed to compute
   page_start / page_end metadata for any arbitrary text span later).
2. Try to detect section headings (numbered headings, markdown headings,
   ALL-CAPS short lines, Title Case short lines). If enough headings are
   found, chunk section-by-section, merging small sections together and
   splitting oversized ones so every parent lands in the configured token
   range.
3. If no reliable headings are found, fall back to plain token-based
   splitting via LangChain's RecursiveCharacterTextSplitter.

Each ParentChunk also carries an internal `start_char` / `end_char` offset
(not persisted to parents.json) so the child chunker can re-derive accurate
page numbers for child chunks without re-scanning the PDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.document_normalizer import NormalizedPage
from utils.helper import count_tokens, get_logger, new_id

logger = get_logger(__name__)

# Rough chars-per-token used only to size the character-based splitter;
# actual token counts are always re-verified with count_tokens().
_CHARS_PER_TOKEN = 4

_HEADING_PATTERNS = [
    re.compile(r"^\s{0,3}#{1,6}\s+\S.*$"),                      # markdown headings
    re.compile(r"^\s{0,3}(\d{1,2}(\.\d{1,2}){0,3})[\).]?\s+\S.*$"),  # 1. / 1.1 / 1.1.2 Heading
    re.compile(r"^\s{0,3}(chapter|section|appendix)\s+\w+.*$", re.IGNORECASE),
]


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    for pattern in _HEADING_PATTERNS:
        if pattern.match(stripped):
            return True
    # ALL CAPS short line, e.g. "INTRODUCTION"
    letters = [c for c in stripped if c.isalpha()]
    if letters and stripped.isupper() and 3 <= len(stripped) <= 80:
        return True
    # Title Case short line with no trailing punctuation, e.g. "System Architecture"
    if (
        3 <= len(stripped) <= 80
        and not stripped.endswith((".", ",", ";", ":"))
        and stripped[0].isupper()
        and len(stripped.split()) <= 10
        and sum(1 for w in stripped.split() if w[:1].isupper()) >= max(1, len(stripped.split()) - 2)
    ):
        return True
    return False


@dataclass
class PageOffset:
    page_number: int
    start: int
    end: int  # exclusive


@dataclass
class ParentChunk:
    parent_id: str
    title: str
    document_name: str
    page_start: int
    page_end: int
    text: str
    token_count: int
    # internal-only, used by child_chunker, not written to parents.json
    start_char: int = field(default=0, repr=False)
    end_char: int = field(default=0, repr=False)

    def to_storage_dict(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "title": self.title,
            "document_name": self.document_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "token_count": self.token_count,
        }


class ParentChunker:
    def __init__(
        self,
        document_name: str,
        min_tokens: int = 1000,
        max_tokens: int = 2000,
    ) -> None:
        self.document_name = document_name
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chunk(self, pages: List[NormalizedPage]) -> Tuple[List[ParentChunk], List[PageOffset]]:
        usable_pages = [p for p in pages if p.text.strip()]
        if not usable_pages:
            logger.warning("No non-empty pages to chunk for document '%s'.", self.document_name)
            return [], []

        full_text, page_offsets = self._build_full_text(usable_pages)
        heading_offsets = self._find_heading_offsets(full_text)

        if len(heading_offsets) >= 2:
            logger.info(
                "Detected %d section headings in '%s' -> using section-based parent chunking.",
                len(heading_offsets),
                self.document_name,
            )
            spans = self._section_based_spans(full_text, heading_offsets)
        else:
            logger.info(
                "No reliable headings detected in '%s' -> falling back to token-based parent chunking.",
                self.document_name,
            )
            spans = self._token_based_spans(full_text)

        parents = [
            self._build_parent_chunk(full_text, start, end, title, page_offsets)
            for start, end, title in spans
            if full_text[start:end].strip()
        ]
        logger.info("Built %d parent chunks for '%s'.", len(parents), self.document_name)
        return parents, page_offsets

    # ------------------------------------------------------------------ #
    # Full-text + page-offset construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_full_text(pages: List[NormalizedPage]) -> Tuple[str, List[PageOffset]]:
        parts: List[str] = []
        offsets: List[PageOffset] = []
        cursor = 0
        separator = "\n\n"

        for page in pages:
            start = cursor
            parts.append(page.text)
            cursor += len(page.text)
            offsets.append(PageOffset(page_number=page.page_number, start=start, end=cursor))
            parts.append(separator)
            cursor += len(separator)

        return "".join(parts), offsets

    @staticmethod
    def _pages_for_span(start: int, end: int, page_offsets: List[PageOffset]) -> Tuple[int, int]:
        overlapping = [po.page_number for po in page_offsets if po.start < end and po.end > start]
        if not overlapping:
            # fall back to nearest page if rounding left no exact overlap
            nearest = min(page_offsets, key=lambda po: abs(po.start - start))
            return nearest.page_number, nearest.page_number
        return min(overlapping), max(overlapping)

    # ------------------------------------------------------------------ #
    # Heading detection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_heading_offsets(full_text: str) -> List[Tuple[int, str]]:
        """Return list of (char_offset, heading_text) for detected heading lines."""
        offsets: List[Tuple[int, str]] = []
        cursor = 0
        for line in full_text.split("\n"):
            if _looks_like_heading(line):
                offsets.append((cursor, line.strip()))
            cursor += len(line) + 1  # +1 for the '\n' we split on
        return offsets

    # ------------------------------------------------------------------ #
    # Section-based spans
    # ------------------------------------------------------------------ #
    def _section_based_spans(
        self, full_text: str, heading_offsets: List[Tuple[int, str]]
    ) -> List[Tuple[int, int, Optional[str]]]:
        boundaries = [offset for offset, _ in heading_offsets] + [len(full_text)]
        raw_sections: List[Tuple[int, int, str]] = []
        for i in range(len(heading_offsets)):
            start = heading_offsets[i][0]
            end = boundaries[i + 1]
            raw_sections.append((start, end, heading_offsets[i][1]))

        # Leading content before the first heading, if any, becomes its own section.
        if heading_offsets[0][0] > 0:
            raw_sections.insert(0, (0, heading_offsets[0][0], "Preamble"))

        merged: List[Tuple[int, int, str]] = []
        buffer_start, buffer_end, buffer_title = raw_sections[0]
        for start, end, title in raw_sections[1:]:
            buffer_tokens = count_tokens(full_text[buffer_start:buffer_end])
            if buffer_tokens < self.min_tokens:
                # merge current section into the buffer
                buffer_end = end
            else:
                merged.append((buffer_start, buffer_end, buffer_title))
                buffer_start, buffer_end, buffer_title = start, end, title
        merged.append((buffer_start, buffer_end, buffer_title))

        # Split any oversized sections using the token-based splitter.
        final_spans: List[Tuple[int, int, Optional[str]]] = []
        for start, end, title in merged:
            section_tokens = count_tokens(full_text[start:end])
            if section_tokens <= self.max_tokens:
                final_spans.append((start, end, title))
                continue
            for sub_start, sub_end in self._split_range_token_based(full_text, start, end):
                final_spans.append((sub_start, sub_end, title))

        return final_spans

    # ------------------------------------------------------------------ #
    # Token-based fallback (also used to split oversized sections)
    # ------------------------------------------------------------------ #
    def _token_based_spans(self, full_text: str) -> List[Tuple[int, int, Optional[str]]]:
        return [(s, e, None) for s, e in self._split_range_token_based(full_text, 0, len(full_text))]

    def _split_range_token_based(self, full_text: str, range_start: int, range_end: int) -> List[Tuple[int, int]]:
        target_chars = int(((self.min_tokens + self.max_tokens) / 2) * _CHARS_PER_TOKEN)
        overlap_chars = 0  # parents are stored independently; no overlap needed between them
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        sub_text = full_text[range_start:range_end]
        pieces = splitter.split_text(sub_text)

        spans: List[Tuple[int, int]] = []
        cursor = range_start
        for piece in pieces:
            idx = full_text.find(piece, cursor, range_end)
            if idx == -1:
                # Splitter normalized whitespace; fall back to sequential placement.
                idx = cursor
            start = idx
            end = idx + len(piece)
            spans.append((start, end))
            cursor = end
        return spans

    # ------------------------------------------------------------------ #
    # Parent chunk construction
    # ------------------------------------------------------------------ #
    def _build_parent_chunk(
        self,
        full_text: str,
        start: int,
        end: int,
        title: Optional[str],
        page_offsets: List[PageOffset],
    ) -> ParentChunk:
        text = full_text[start:end].strip()
        page_start, page_end = self._pages_for_span(start, end, page_offsets)
        return ParentChunk(
            parent_id=new_id("parent_"),
            title=title or f"{self.document_name} (p.{page_start}-{page_end})",
            document_name=self.document_name,
            page_start=page_start,
            page_end=page_end,
            text=text,
            token_count=count_tokens(text),
            start_char=start,
            end_char=end,
        )
