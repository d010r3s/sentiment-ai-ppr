# src/data/database.py
import sqlite3


def init_db(db_path="data/feedback.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT,
            comment TEXT,
            aspect TEXT,
            tone TEXT,
            embedding BLOB,
            recommendations TEXT
        )
    """)
    # Add columns if they don't exist
    cursor.execute("PRAGMA table_info(feedback)")
    columns = [col[1] for col in cursor.fetchall()]
    if "aspect" not in columns:
        cursor.execute("ALTER TABLE feedback ADD COLUMN aspect TEXT")
    if "embedding" not in columns:
        cursor.execute("ALTER TABLE feedback ADD COLUMN embedding BLOB")
    if "recommendations" not in columns:
        cursor.execute("ALTER TABLE feedback ADD COLUMN recommendations TEXT")
    conn.commit()
    conn.close()