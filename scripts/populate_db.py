import pandas as pd
import sqlite3
import pickle
import re

from src.models.embedder import Embedder
from src.models.recommender import Recommender
from src.data.database import init_db
from src.data.preprocess import preprocess_data
from src.utils.config import load_config


def is_valid_recommendation(rec):
    """
    Check if recommendation is valid:
    - Non-empty string
    - Matches numbered list pattern (e.g. "1. ...\n2. ...")
    - Has between 4 and 6 lines (recommendations)
    """
    if not rec or not isinstance(rec, str):
        return False
    pattern = r"^\d+\..*\n.*\d+\..*$"
    return bool(re.match(pattern, rec, re.MULTILINE)) and 4 <= len(rec.split("\n")) <= 6


def populate_db():
    config = load_config()
    db_path = config["database"]["path"]
    model_id = config["models"]["recommender"]
    process_neutral = config["processing"]["include_neutral"]
    use_fallback = config["processing"]["use_fallback_data"]

    # Initialize database
    init_db(db_path)

    # Preprocess data
    preprocess_data()  # Generates data/preprocessed_reviews.csv

    # Initialize sentiments
    sentiments = ["negative"]
    if process_neutral:
        sentiments.append("neutral")

    # Load preprocessed comments CSV
    try:
        df = pd.read_csv("data/preprocessed_reviews.csv", sep=';', encoding="cp1252")
        comments = df.loc[df["sentiment"].isin(sentiments), ["text", "sentiment", "comment_id"]].copy()
        comments.rename(columns={"text": "comment", "sentiment": "tone"}, inplace=True)
    except FileNotFoundError:
        if not use_fallback:
            raise FileNotFoundError("data/preprocessed_reviews.csv not found. Set 'use_fallback_data: true' \
                                    in config.yaml.")
        print("Error: data/preprocessed_reviews.csv not found. Using sample data.")
        comments = pd.DataFrame([
            {"comment": "Долго возвращают средства", "tone": "negative", "comment_id": "f4b8b..."},  # Precomputed hash
            {"comment": "Приложение постоянно крашится", "tone": "negative", "comment_id": "a9c2d..."},
            {"comment": "Поддержка работает окей", "tone": "neutral", "comment_id": "e7f1b..."}
        ])
        comments = comments[comments["tone"].isin(sentiments)]

    # Generate recommendations with Recommender class
    recommender = Recommender(model_id=model_id, retriever=None)
    recommendations = recommender.generate_batch(comments["comment"].tolist(), batch_size=4)
    comments["recommendations"] = recommendations

    # Filter valid recommendations only
    comments["is_valid"] = comments["recommendations"].apply(is_valid_recommendation)
    valid_comments = comments[comments["is_valid"]].drop(columns=["is_valid"])
    invalid_count = len(comments) - len(valid_comments)
    if invalid_count > 0:
        print(f"Warning: {invalid_count} invalid recommendations filtered out.")

    # Save valid recommendations to CSV
    csv_path = "data/recommendations.csv"
    valid_comments.to_csv(csv_path, index=False, sep=';', encoding="cp1252")
    print(f"Saved {len(valid_comments)} recommendations to {csv_path}")

    # Generate embeddings with Embedder
    embedder = Embedder()
    valid_comments["embedding"] = valid_comments["comment"].apply(lambda x: pickle.dumps(embedder.encode(x)))

    # Insert records into the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for _, row in valid_comments.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO feedback (comment, tone, aspect, recommendations, embedding, comment_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (row["comment"], row.get("aspect", None), row["tone"], row["recommendations"], row["embedding"],
              row["comment_id"]))
    conn.commit()
    conn.close()
    print(f"Inserted {len(valid_comments)} records into {db_path}")


if __name__ == "__main__":
    populate_db()