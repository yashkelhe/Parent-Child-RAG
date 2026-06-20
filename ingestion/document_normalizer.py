"""
ingestion/document_normalizer.py

Cleans raw per-page text:
- collapses repeated whitespace / blank lines
- detects and strips repeating headers/footers (lines that recur across
  most pages, e.g. "Confidential | Company X" or page numbers)
- drops pages that are empty after cleaning, while logging the fact
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List

from ingestion.pdf_loader import PageContent
from utils.helper import get_logger

logger = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*/\s*\d{1,4})?\s*$", re.IGNORECASE)


@dataclass
class NormalizedPage:
    page_number: int
    text: str


class DocumentNormalizer:
    """Removes boilerplate (headers/footers) and normalizes whitespace."""

    def __init__(self, header_footer_threshold: float = 0.5) -> None:
        """
        header_footer_threshold: fraction of pages a line must appear on
        (after light normalization) to be considered a repeating
        header/footer and stripped out.
        """
        self.header_footer_threshold = header_footer_threshold

    def normalize(self, pages: List[PageContent]) -> List[NormalizedPage]:
        if not pages:
            return []

        candidate_lines = self._find_repeating_lines(pages)

        normalized: List[NormalizedPage] = []
        for page in pages:
            cleaned = self._clean_page(page.text, candidate_lines)
            if not cleaned.strip():
                logger.debug("Page %d is empty after normalization; keeping as blank.", page.page_number)
            normalized.append(NormalizedPage(page_number=page.page_number, text=cleaned))

        logger.info(
            "Normalized %d pages (%d repeating header/footer lines removed).",
            len(pages),
            len(candidate_lines),
        )
        return normalized

    def _find_repeating_lines(self, pages: List[PageContent]) -> set:
        """Identify lines that recur on a large fraction of pages — likely headers/footers."""
        line_page_counts: Counter = Counter()
        total_pages = len(pages)

        for page in pages:
            lines = {self._normalize_line(l) for l in page.text.splitlines() if l.strip()}
            for line in lines:
                if line:
                    line_page_counts[line] += 1

        threshold_count = max(2, int(total_pages * self.header_footer_threshold))
        repeating = {
            line
            for line, count in line_page_counts.items()
            if count >= threshold_count and len(line) < 120  # headers/footers are usually short
        }
        return repeating

    @staticmethod
    def _normalize_line(line: str) -> str:
        line = line.strip().lower()
        # collapse digits so "Page 3" and "Page 4" are treated as the same template
        line = re.sub(r"\d+", "#", line)
        return line

    def _clean_page(self, text: str, repeating_lines: set) -> str:
        out_lines = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                out_lines.append("")
                continue
            if self._normalize_line(stripped) in repeating_lines:
                continue
            if _PAGE_NUMBER_LINE_RE.match(stripped):
                continue
            out_lines.append(_WHITESPACE_RE.sub(" ", stripped))

        joined = "\n".join(out_lines)
        joined = _MULTI_BLANK_RE.sub("\n\n", joined)
        return joined.strip()
