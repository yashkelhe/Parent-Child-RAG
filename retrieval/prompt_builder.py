"""
retrieval/prompt_builder.py

Builds the final prompt sent to Gemini: retrieved parent context + the
user's question + a strict instruction to answer only from that context.
"""

from __future__ import annotations

from typing import List

from ingestion.parent_chunker import ParentChunk

_SYSTEM_INSTRUCTION = (
    "You are a precise, factual assistant. Answer the user's question using "
    "ONLY the context provided below. Do not use outside knowledge. "
    "If the answer is not contained in the context, say clearly that the "
    "provided documents do not contain enough information to answer. "
    "Cite the source document and page range for any claim you make, using "
    "the format (source: <document_name>, p.<page_start>-<page_end>)."
)


class PromptBuilder:
    @staticmethod
    def build(question: str, parents: List[ParentChunk]) -> str:
        if not parents:
            context_block = "No relevant context was found."
        else:
            sections = []
            for parent in parents:
                sections.append(
                    f"---\n"
                    f"Source: {parent.document_name} | Section: {parent.title} | "
                    f"Pages: {parent.page_start}-{parent.page_end}\n"
                    f"{parent.text}\n"
                )
            context_block = "\n".join(sections)

        prompt = (
            f"{_SYSTEM_INSTRUCTION}\n\n"
            f"=== CONTEXT START ===\n"
            f"{context_block}\n"
            f"=== CONTEXT END ===\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return prompt
