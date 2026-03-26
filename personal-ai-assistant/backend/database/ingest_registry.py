"""Postgres registry for idempotent per-user document ingest (content hash)."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS user_document_ingest (
    user_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, content_hash)
);
"""


def ensure_ingest_table(postgres_uri: str) -> None:
    """Create user_document_ingest if missing."""
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not available; ingest registry disabled")
        return
    try:
        conn = psycopg2.connect(postgres_uri)
        try:
            with conn.cursor() as cur:
                cur.execute(_CREATE_SQL)
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Could not ensure user_document_ingest table: %s", e)


def is_ingested(postgres_uri: str, user_id: str, content_hash: str) -> bool:
    """Return True if this user+hash was already ingested."""
    try:
        import psycopg2
    except ImportError:
        return False
    try:
        conn = psycopg2.connect(postgres_uri)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM user_document_ingest WHERE user_id = %s AND content_hash = %s",
                    (user_id, content_hash),
                )
                return cur.fetchone() is not None
        finally:
            conn.close()
    except Exception:
        return False


def mark_ingested(postgres_uri: str, user_id: str, content_hash: str) -> None:
    """Record a successful ingest."""
    try:
        import psycopg2
    except ImportError:
        return
    conn = psycopg2.connect(postgres_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_document_ingest (user_id, content_hash)
                VALUES (%s, %s)
                ON CONFLICT (user_id, content_hash) DO NOTHING
                """,
                (user_id, content_hash),
            )
        conn.commit()
    finally:
        conn.close()
