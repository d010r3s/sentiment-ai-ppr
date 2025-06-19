import pandas as pd
import sqlite3
import pickle
import yaml
import re
import os
import sys

from sentence_transformers import SentenceTransformer
from src.models.recommender import Recommender
from src.data.database import init_db
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
    embedder_model = config["models"]["embedder"]
    process_neutral = config.get("processing", {}).get("include_neutral", False)

    # Initialize database
    init_db(db_path)

    # Initialize sentiments here so it's defined in all branches
    sentiments = ["negative"]
    if process_neutral:
        sentiments.append("neutral")

    # Load comments CSV
    try:
        df = pd.read_csv("data/all_reviews.csv", sep=';', encoding="cp1252")
        comments = df.loc[df["sentiment"].isin(sentiments), ["text", "sentiment"]].copy()
        comments.rename(columns={"text": "comment", "sentiment": "tone"}, inplace=True)
    except FileNotFoundError:
        print("Error: data/all_reviews.csv not found. Using sample data.")
        comments = pd.DataFrame([
            {"comment": "Долго возвращают средства", "tone": "negative"},
            {"comment": "Приложение постоянно крашится", "tone": "negative"},
            {"comment": "Поддержка работает окей", "tone": "neutral"}
        ])
        comments = comments[comments["tone"].isin(sentiments)]

    # Generate recommendations with your Recommender class
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
    valid_comments.to_csv(csv_path, index=False)
    print(f"Saved {len(valid_comments)} recommendations to {csv_path}")

    # Generate embeddings with SentenceTransformer
    embedder = SentenceTransformer(embedder_model)
    valid_comments["embedding"] = valid_comments["comment"].apply(lambda x: pickle.dumps(embedder.encode(x)))

    # Insert records into the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for _, row in valid_comments.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO feedback (comment, tone, aspect, recommendations, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (row["comment"], row.get("aspect", None), row["tone"], row["recommendations"], row["embedding"]))
    conn.commit()
    conn.close()
    print(f"Inserted {len(valid_comments)} records into {db_path}")


if __name__ == "__main__":
    populate_db()
