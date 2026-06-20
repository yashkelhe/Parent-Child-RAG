"""
ingestion/pdf_loader.py

Loads a PDF using PyMuPDF (fitz) and extracts per-page text. Handles
malformed PDFs and empty pages gracefully instead of crashing the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from utils.helper import get_logger

logger = get_logger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be opened or read at all."""


@dataclass
class PageContent:
    """A single page of raw extracted text."""

    page_number: int  # 1-indexed
    text: str


class PDFLoader:
    """Responsible solely for turning a PDF file into raw per-page text."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(self) -> List[PageContent]:
        """
        Extract text from every page of the PDF.

        Returns a list of PageContent. Pages that fail to extract or are
        empty are still included (with empty text) so page numbering stays
        consistent for downstream metadata - callers can filter them out.
        """
        if not self.file_path.exists():
            raise PDFLoadError(f"PDF not found at path: {self.file_path}")

        try:
            doc = fitz.open(self.file_path)
        except Exception as exc:
            raise PDFLoadError(f"Could not open PDF '{self.file_path.name}': {exc}") from exc

        if doc.is_encrypted:
            # Try an empty-password unlock; many "protected" PDFs allow read access.
            if not doc.authenticate(""):
                doc.close()
                raise PDFLoadError(f"PDF '{self.file_path.name}' is encrypted and could not be opened.")

        pages: List[PageContent] = []
        for index in range(doc.page_count):
            page_number = index + 1
            try:
                page = doc.load_page(index)
                text = page.get_text("text") or ""
            except Exception as exc:
                logger.warning(
                    "Failed to extract text from page %d of '%s': %s. Treating as empty page.",
                    page_number,
                    self.file_path.name,
                    exc,
                )
                text = ""

            if not text.strip():
                logger.debug("Page %d of '%s' is empty.", page_number, self.file_path.name)

            pages.append(PageContent(page_number=page_number, text=text))

        doc.close()

        if not pages:
            raise PDFLoadError(f"PDF '{self.file_path.name}' contains no pages.")

        non_empty = sum(1 for p in pages if p.text.strip())
        logger.info(
            "Loaded '%s': %d pages total, %d with extractable text.",
            self.file_path.name,
            len(pages),
            non_empty,
        )
        return pages
