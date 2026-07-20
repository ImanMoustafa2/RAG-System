"""
Persistent query -> answer cache backed by SQLite.

Caches the full pipeline output (retrieved chunk ids + final answer) keyed
by a hash of (query text, active metadata filters). Saves cost/latency for
repeated or near-identical questions, which is common in a lab setting
(e.g. multiple students asking "flash point of ethanol?").
"""
import hashlib
import json
import sqlite3
import time
from typing import Any, Dict, Optional

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_cache (
    cache_key TEXT PRIMARY KEY,
    query TEXT,
    payload TEXT,
    created_at REAL
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.QUERY_CACHE_DB)
    conn.execute(_SCHEMA)
    return conn


def make_cache_key(query: str, filters: Optional[Dict[str, str]] = None) -> str:
    payload = json.dumps({"query": query.strip().lower(), "filters": filters or {}}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached(cache_key: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT payload, created_at FROM query_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if not row:
            return None
        payload, created_at = row
        if time.time() - created_at > config.CACHE_TTL_SECONDS:
            return None
        return json.loads(payload)
    finally:
        conn.close()


def set_cached(cache_key: str, query: str, payload: Dict[str, Any]) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO query_cache (cache_key, query, payload, created_at) VALUES (?, ?, ?, ?)",
            (cache_key, query, json.dumps(payload), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def clear_cache() -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM query_cache")
        conn.commit()
    finally:
        conn.close()
