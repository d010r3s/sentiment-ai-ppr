# src/data/database.py
import sqlite3
from src.utils.config import load_config


def init_db(db_path=None):
    """
    Initialize SQLite database with feedback table.
    Args:
        db_path (str, optional): Path to SQLite database.
    """
    config = load_config()
    db_path = db_path or config["database"]["path"]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT,
            comment TEXT,
            tone TEXT,
            embedding BLOB,
            recommendations TEXT,
            aspect TEXT,
            comment_id TEXT UNIQUE
        )
    """)
    # Add columns if they don't exist
    cursor.execute("PRAGMA table_info(feedback)")
    columns = [col[1] for col in cursor.fetchall()]
    if "embedding" not in columns:
        cursor.execute("ALTER TABLE feedback ADD COLUMN embedding BLOB")
    if "recommendations" not in columns:
        cursor.execute("ALTER TABLE feedback ADD COLUMN recommendations TEXT")
    if "aspect" not in columns:
        cursor.execute("ALTER TABLE feedback ADD COLUMN aspect TEXT")
    if "comment_id" not in columns:
        cursor.execute("ALTER TABLE feedback ADD COLUMN comment_id TEXT UNIQUE")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()