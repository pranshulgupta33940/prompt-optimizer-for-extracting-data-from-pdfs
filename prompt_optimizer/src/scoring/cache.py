"""SQLite-backed caches for stochastic metrics and LLM extraction results.

Two caches live here:

* **MetricCache** — keyed by ``(metric_id, predicted, gold)`` so that
  LLM-based scoring calls (``string_semantic``, ``array_llm``) are never
  repeated for the same pair.
* **ExtractionCache** — keyed by ``(document_path, prompt_hash)`` so that
  extraction calls are never repeated after a crash / restart.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


class MetricCache:
    """Persistent cache for stochastic metric results (SQLite, WAL mode)."""

    _CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS metric_cache (
            key   TEXT PRIMARY KEY,
            score REAL NOT NULL,
            reason TEXT NOT NULL
        )
    """

    def __init__(self, cache_path: Path) -> None:
        """Open (or create) the SQLite cache at *cache_path*.

        Args:
            cache_path: Filesystem path for the ``.db`` file.
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(cache_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(self._CREATE_SQL)
        self._conn.commit()

    def get(
        self, metric_id: str, predicted: Any, gold: Any,
    ) -> tuple[float, str] | None:
        """Look up a cached result.

        Returns:
            ``(score, reason)`` if cached, otherwise ``None``.
        """
        key = self._make_key(metric_id, predicted, gold)
        row = self._conn.execute(
            "SELECT score, reason FROM metric_cache WHERE key = ?", (key,),
        ).fetchone()
        if row is not None:
            return float(row[0]), str(row[1])
        return None

    def put(
        self,
        metric_id: str,
        predicted: Any,
        gold: Any,
        score: float,
        reason: str,
    ) -> None:
        """Store a result in the cache (upsert)."""
        key = self._make_key(metric_id, predicted, gold)
        self._conn.execute(
            "INSERT OR REPLACE INTO metric_cache (key, score, reason) "
            "VALUES (?, ?, ?)",
            (key, score, reason),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    @staticmethod
    def _make_key(metric_id: str, predicted: Any, gold: Any) -> str:
        """Generate a deterministic SHA-256 cache key."""
        payload = json.dumps(
            {"m": metric_id, "p": predicted, "g": gold},
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class ExtractionCache:
    """Persistent cache for LLM extraction results (SQLite, WAL mode)."""

    _CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS extraction_cache (
            key       TEXT PRIMARY KEY,
            result    TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """

    def __init__(self, cache_path: Path) -> None:
        """Open (or create) the extraction cache at *cache_path*.

        Args:
            cache_path: Filesystem path for the ``.db`` file.
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(cache_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(self._CREATE_SQL)
        self._conn.commit()

    def get(self, doc_path: Path, prompt: str) -> dict | None:
        """Look up a cached extraction result.

        Args:
            doc_path: Path to the source PDF.
            prompt: The extraction prompt that was used.

        Returns:
            Parsed JSON dict if cached, otherwise ``None``.
        """
        key = self._make_key(doc_path, prompt)
        row = self._conn.execute(
            "SELECT result FROM extraction_cache WHERE key = ?", (key,),
        ).fetchone()
        if row is not None:
            return json.loads(row[0])
        return None

    def put(
        self,
        doc_path: Path,
        prompt: str,
        result: dict,
        timestamp: str = "",
    ) -> None:
        """Store an extraction result in the cache."""
        import datetime

        key = self._make_key(doc_path, prompt)
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO extraction_cache (key, result, timestamp) "
            "VALUES (?, ?, ?)",
            (key, json.dumps(result, default=str), ts),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    @staticmethod
    def _make_key(doc_path: Path, prompt: str) -> str:
        """Generate a deterministic SHA-256 cache key."""
        payload = f"{doc_path.resolve()}||{prompt}"
        return hashlib.sha256(payload.encode()).hexdigest()
