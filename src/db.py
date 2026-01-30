"""
Database connection utilities for FAS Smart.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Generator

from config import config


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Context manager for database connections.
    
    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    conn = psycopg2.connect(**config.DB.get_dict())
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_cursor(dict_cursor: bool = False) -> Generator[psycopg2.extensions.cursor, None, None]:
    """
    Context manager for database cursors with auto-commit.
    
    Usage:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM trading_pairs")
            rows = cur.fetchall()
    """
    cursor_factory = RealDictCursor if dict_cursor else None
    
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def test_connection() -> bool:
    """Test database connection"""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1
    except Exception as e:
        print(f"Connection failed: {e}")
        return False
