import pandas as pd
import sqlite3
import pickle
from sentence_transformers import SentenceTransformer
from src.data.database import init_db
import yaml
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def is_valid_recommendation(rec):
    """Check if recommendation is valid (non-empty, numbered list format)."""
    if not rec or not isinstance(rec, str):
        return False
    pattern = r"^\d+\..*\n.*\d+\..*$"
    return bool(re.match(pattern, rec, re.MULTILINE)) and 4 <= len(rec.split("\n")) <= 6


def extract_aspect(comment):
    """Placeholder for aspect extraction model (replace with actual model)."""
    if "refund" in comment.lower():
        return "refund speed"
    elif "app" in comment.lower() or "crash" in comment.lower():
        return "app stability"
    elif "support" in comment.lower():
        return "support quality"
    else:
        return "general"


def get_prewritten_recommendations(aspect):
    """Return pre-written recommendations based on aspect."""
    recommendations = {
        "refund speed": """1. Reduce refund processing time to 5 business days.
2. Implement automated refund tracking system.
3. Train staff on efficient refund procedures.
4. Provide real-time refund status updates to customers.""",
        "app stability": """1. Conduct thorough app testing before updates.
2. Fix known crash issues in the next release.
3. Optimize app performance for low-end devices.
4. Offer users a feedback channel for crash reports.""",
        "support quality": """1. Train support team on empathy and problem-solving.
2. Reduce support ticket response time to 24 hours.
3. Implement a 24/7 live chat support option.
4. Create a detailed FAQ for common issues.""",
        "general": """1. Improve overall customer communication.
2. Gather regular feedback through surveys.
3. Enhance user interface for better experience.
4. Offer loyalty discounts to retain customers."""
    }
    return recommendations.get(aspect, recommendations["general"])


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

    # Extract aspects
    comments["aspect"] = comments["comment"].apply(extract_aspect)

    # Assign prewritten recommendations
    comments["recommendations"] = comments["aspect"].apply(get_prewritten_recommendations)

    # Preprocess: Filter valid recommendations
    comments["is_valid"] = comments["recommendations"].apply(is_valid_recommendation)
    valid_comments = comments[comments["is_valid"]].drop(columns=["is_valid"])
    invalid_count = len(comments) - len(valid_comments)
    if invalid_count > 0:
        print(f"Warning: {invalid_count} invalid recommendations filtered out.")

    # Save to CSV
    csv_path = "data/recommendations.csv"
    valid_comments.to_csv(csv_path, index=False)
    print(f"Saved {len(valid_comments)} recommendations to {csv_path}")

    # Generate embeddings
    embedder = SentenceTransformer(embedder_model)
    valid_comments["embedding"] = valid_comments["comment"].apply(lambda x: pickle.dumps(embedder.encode(x)))

    # Insert into database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for _, row in valid_comments.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO feedback (comment, tone, aspect, recommendations, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (row["comment"], row["tone"], row["aspect"], row["recommendations"], row["embedding"]))
    conn.commit()
    conn.close()
    print(f"Inserted {len(valid_comments)} records into {db_path}")


if __name__ == "__main__":
    populate_db()


# # scripts/populate_db.py
# import pandas as pd
# import sqlite3
# import pickle
# from sentence_transformers import SentenceTransformer
# from src.models.recommender import Recommender
# from src.data.database import init_db
# import yaml
# import re
#
#
# def load_config(config_path="config/config.yaml"):
#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)
#
#
# def is_valid_recommendation(rec):
#     """Check if recommendation is valid (non-empty, numbered list format)."""
#     if not rec or not isinstance(rec, str):
#         return False
#     pattern = r"^\d+\..*\n.*\d+\..*$"
#     return bool(re.match(pattern, rec, re.MULTILINE)) and 4 <= len(rec.split("\n")) <= 6
#
#
# def extract_aspect(comment):
#     """Placeholder for aspect extraction model (replace with actual model)."""
#     if "refund" in comment.lower():
#         return "refund speed"
#     elif "app" in comment.lower() or "crash" in comment.lower():
#         return "app stability"
#     elif "support" in comment.lower():
#         return "support quality"
#     else:
#         return "general"
#
#
# def populate_db():
#     config = load_config()
#     db_path = config["database"]["path"]
#     model_id = config["models"]["recommender"]
#     embedder_model = config["models"]["embedder"]
#     process_neutral = config.get("processing", {}).get("include_neutral", False)
#
#     # Initialize database
#     init_db(db_path)
#
#     # Load comments
#     try:
#         df = pd.read_csv("data/all_reviews.csv", sep=';', encoding="cp1252")
#         sentiments = ["negative"]
#         if process_neutral:
#             sentiments.append("neutral")
#         comments = df.loc[df["sentiment"].isin(sentiments), ["text", "sentiment"]].copy()
#         comments.rename(columns={"text": "comment", "sentiment": "tone"}, inplace=True)
#     except FileNotFoundError:
#         print("Error: data/all_reviews.csv not found. Using sample data.")
#         comments = pd.DataFrame([
#             {"comments": "Долго возвращают средства", "tone": "negative"},
#             {"comments": "Приложение постоянно крашится", "tone": "negative"},
#             {"comments": "Поддержка работает окей", "tone": "neutral"}
#         ])
#         comments = comments[comments["tone"].isin(sentiments)]
#
#     # Extract aspects
#     comments["aspect"] = comments["comment"].apply(extract_aspect)
#
#     # Generate recommendations
#     recommender = Recommender(model_id=model_id, retriever=None)
#     recommendations = recommender.generate_batch(comments["comment"].tolist(), batch_size=4)
#     comments["recommendations"] = recommendations
#
#     # Preprocess: Filter valid recommendations
#     comments["is_valid"] = comments["recommendations"].apply(is_valid_recommendation)
#     valid_comments = comments[comments["is_valid"]].drop(columns=["is_valid"])
#     invalid_count = len(comments) - len(valid_comments)
#     if invalid_count > 0:
#         print(f"Warning: {invalid_count} invalid recommendations filtered out.")
#
#     # Save to CSV
#     csv_path = "data/recommendations.csv"
#     valid_comments.to_csv(csv_path, index=False)
#     print(f"Saved {len(valid_comments)} recommendations to {csv_path}")
#
#     # Generate embeddings
#     embedder = SentenceTransformer(embedder_model)
#     valid_comments["embedding"] = valid_comments["comment"].apply(lambda x: pickle.dumps(embedder.encode(x)))
#
#     # Insert into database
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     for _, row in valid_comments.iterrows():
#         cursor.execute("""
#             INSERT OR REPLACE INTO feedback (comment, tone, aspect, recommendations, embedding)
#             VALUES (?, ?, ?, ?, ?)
#         """, (row["comment"], row["tone"], row["aspect"], row["recommendations"], row["embedding"]))
#     conn.commit()
#     conn.close()
#     print(f"Inserted {len(valid_comments)} records into {db_path}")
#
#
# if __name__ == "__main__":
#     populate_db()