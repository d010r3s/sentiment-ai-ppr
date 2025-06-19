import pandas as pd
import sqlite3
import re
from typing import Optional, Union

from src.models.recommender import Recommender
from src.data.database import init_db
from src.data.preprocess import preprocess_data
from src.utils.config import load_config


def is_valid_recommendation(rec: Optional[str]) -> bool:
    """
    Check if recommendation is valid:
    - Non-empty string
    - Matches numbered list pattern (e.g. "1. ...\n2. ...")
    - Has between 4 and 6 lines (recommendations)
    Args:
        rec: Recommendation text to validate.
    Returns:
        True if valid, False otherwise.
    """
    if not rec or not isinstance(rec, str):
        return False
    pattern = r"^\d+\..*\n.*\d+\..*$"
    return bool(re.match(pattern, rec, re.MULTILINE)) and 4 <= len(rec.split("\n")) <= 6


def populate_db(input_data: Union[str, pd.DataFrame] = "data/all_reviews.csv", generate_recommendations: bool = False):
    """
    Preprocess reviews and populate the feedback table in the database.
    Optionally generate recommendations.
    Args:
        input_data: Path to input CSV or preprocessed DataFrame.
        generate_recommendations: If True, generate and insert recommendations.
    """
    try:
        config = load_config()
        db_path = config["database"]["path"]
        model_id = config["models"]["recommender"]

        # Initialize database
        init_db(db_path)

        # Preprocess data if input is a CSV path
        df = preprocess_data(input_data) if isinstance(input_data, str) else input_data
        if df.empty:
            print("No preprocessed data to process. Exiting.")
            return

        # Insert preprocessed data into the database
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR IGNORE INTO feedback (comment_id, brand, comment, tone, embedding, recommendations, aspect)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, df[['comment_id', 'brand', 'comment', 'tone', 'embedding', 'recommendations', 'aspect']].values.tolist())
            conn.commit()
            inserted_count = cursor.rowcount  # Note: rowcount may be inaccurate for executemany in SQLite
            print(f"Inserted approximately {inserted_count} preprocessed reviews into the database at {db_path}")

        if not generate_recommendations:
            print("Recommendation generation disabled. Database populated with preprocessed data.")
            return

        # Query comments without recommendations
        with sqlite3.connect(db_path) as conn:
            comments_df = pd.read_sql_query("""
                SELECT comment, tone, comment_id
                FROM feedback
                WHERE recommendations IS NULL
            """, conn)

        if comments_df.empty:
            print("No comments need recommendations. Database is up-to-date.")
            return

        # Generate recommendations
        recommender = Recommender(model_id=model_id, retriever=None)
        recommendations = recommender.generate_batch(comments_df["comment"].tolist(), batch_size=4)
        comments_df["recommendations"] = recommendations

        # Filter valid recommendations
        comments_df["is_valid"] = comments_df["recommendations"].apply(is_valid_recommendation)
        valid_comments = comments_df[comments_df["is_valid"]].copy()  # .copy() prevents SettingWithCopyWarning
        invalid_count = len(comments_df) - len(valid_comments)
        if invalid_count > 0:
            print(f"Warning: {invalid_count} invalid recommendations filtered out.")

        # Update database with recommendations
        if valid_comments.empty:
            print("No valid recommendations to update in the database.")
            return

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                UPDATE feedback
                SET recommendations = ?
                WHERE comment_id = ?
            """, valid_comments[["recommendations", "comment_id"]].values.tolist())
            conn.commit()
            updated_count = len(valid_comments)
            print(f"Updated {updated_count} records with recommendations in {db_path}")

    except Exception as e:
        print(f"Error during database population: {str(e)}")