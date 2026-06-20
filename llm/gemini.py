"""
llm/gemini.py

Thin wrapper around the Gemini generative model (google-genai SDK) used
for final answer synthesis.
"""

from __future__ import annotations

from typing import Optional

from google import genai
from google.genai import types

from config import settings
from utils.helper import get_logger

logger = get_logger(__name__)


class LLMError(Exception):
    """Raised when the Gemini LLM call fails irrecoverably."""


class GeminiLLM:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or settings.gemini.api_key
        self.model_name = model or settings.gemini.llm_model
        self.temperature = temperature if temperature is not None else settings.gemini.temperature
        self.max_output_tokens = max_output_tokens or settings.gemini.max_output_tokens

        if not self.api_key:
            raise LLMError("GOOGLE_API_KEY is not set. Cannot initialize Gemini LLM.")

        self._client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise LLMError("Cannot generate an answer for an empty prompt.")

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
        except Exception as exc:
            raise LLMError(f"Gemini generation call failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            logger.warning("Gemini returned an empty response.")
            return "I could not generate an answer from the provided context."
        return text.strip()
