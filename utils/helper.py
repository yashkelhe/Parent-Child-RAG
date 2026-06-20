"""
utils/helper.py

Shared, dependency-light helpers used across the pipeline:
- logging setup
- token counting (tiktoken, with a safe fallback)
- JSON read/write helpers
- id generation
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, List

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken should normally be installed
    _ENCODER = None


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger that writes to stdout, idempotently."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


def count_tokens(text: str) -> int:
    """
    Approximate token count for a string.

    Uses tiktoken's cl100k_base encoding as a stable, fast proxy for token
    count. This will not be byte-identical to Gemini's own tokenizer, but is
    consistent and good enough for chunk-size budgeting.
    """
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    # Fallback: rough heuristic (~4 chars per token for English prose)
    return max(1, len(text) // 4)


def new_id(prefix: str = "") -> str:
    """Generate a short, unique id, optionally prefixed."""
    token = uuid.uuid4().hex
    return f"{prefix}{token}" if prefix else token


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from disk, returning `default` if the file doesn't exist."""
    if not path.exists():
        return default if default is not None else []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise IOError(f"Failed to read JSON file at {path}: {exc}") from exc


def save_json(path: Path, data: Any) -> None:
    """Write JSON to disk, creating parent directories as needed."""
    ensure_parent_dir(path)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise IOError(f"Failed to write JSON file at {path}: {exc}") from exc


def chunked(iterable: List[Any], size: int) -> List[List[Any]]:
    """Split a list into batches of at most `size` elements."""
    if size <= 0:
        raise ValueError("Batch size must be a positive integer")
    return [iterable[i : i + size] for i in range(0, len(iterable), size)]
