# src/data/database.py
import sqlite3
from src.utils.config import load_config
from typing import Optional


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize SQLite database with feedback table.
    Args:
        db_path: Path to SQLite database. Defaults to path from config.
    """
    config = load_config()
    db_path = db_path or config["database"]["path"]
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                comment_id TEXT PRIMARY KEY,
                brand TEXT,
                comment TEXT,
                tone TEXT,
                rating INTEGER,
                embedding BLOB,
                recommendations TEXT,
                aspect TEXT
            )
        """)
        conn.commit()