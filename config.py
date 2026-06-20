"""
config.py

Centralized, config-driven settings for the Parent-Child RAG system.
All tunable parameters are loaded from environment variables (.env) so
nothing is hardcoded deep inside the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"))
    embed_model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"))
    embedding_dim: int = field(default_factory=lambda: _get_int("GEMINI_EMBED_DIM", 3072 ))
    temperature: float = field(default_factory=lambda: _get_float("GEMINI_TEMPERATURE", 0.2))
    max_output_tokens: int = field(default_factory=lambda: _get_int("GEMINI_MAX_OUTPUT_TOKENS", 1024))


@dataclass(frozen=True)
class PineconeConfig:
    api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "parent-child-rag"))
    cloud: str = field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"))
    region: str = field(default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1"))
    metric: str = field(default_factory=lambda: os.getenv("PINECONE_METRIC", "cosine"))


@dataclass(frozen=True)
class ChunkingConfig:
    parent_min_tokens: int = field(default_factory=lambda: _get_int("PARENT_CHUNK_MIN_TOKENS", 1000))
    parent_max_tokens: int = field(default_factory=lambda: _get_int("PARENT_CHUNK_MAX_TOKENS", 2000))
    child_min_tokens: int = field(default_factory=lambda: _get_int("CHILD_CHUNK_MIN_TOKENS", 250))
    child_max_tokens: int = field(default_factory=lambda: _get_int("CHILD_CHUNK_MAX_TOKENS", 400))
    child_overlap_tokens: int = field(default_factory=lambda: _get_int("CHILD_CHUNK_OVERLAP_TOKENS", 50))


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_children: int = field(default_factory=lambda: _get_int("TOP_K_CHILDREN", 8))
    max_parent_contexts: int = field(default_factory=lambda: _get_int("MAX_PARENT_CONTEXTS", 4))


@dataclass(frozen=True)
class StorageConfig:
    parents_json_path: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("PARENTS_JSON_PATH", "storage/parents.json")
    )
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")


@dataclass(frozen=True)
class Settings:
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()
