# tests/test_database.py
import pytest
import sqlite3
import os
from src.data.database import init_db


@pytest.fixture
def temp_db(tmp_path):
    """Fixture that provides a temporary database path"""
    db_path = tmp_path / "test.db"
    yield str(db_path)
    if os.path.exists(db_path):
        os.remove(db_path)


def test_db_init(temp_db):
    # Initialize the DB
    init_db(temp_db)

    # Verify schema
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
        assert cursor.fetchone()[0] == "feedback"

        # Check columns
        cursor.execute("PRAGMA table_info(feedback)")
        columns = [col[1] for col in cursor.fetchall()]
        expected_columns = {"comment_id", "brand", "comment", "tone", "rating"}
        assert expected_columns.issubset(columns)