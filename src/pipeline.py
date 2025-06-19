# # Orchestrates scraper, preprocessor, RAG, recommender
# # src/pipeline.py
import pandas as pd
import sqlite3
from src.data.database import init_db
from src.rag.retriever import Retriever
from src.models.recommender import Recommender
from src.utils.config import load_config


def main():
    config = load_config()
    db_path = config["database"]["path"]
    process_neutral = config.get("processing", {}).get("include_neutral", False)

    # Initialize database
    init_db(db_path)

    # Load comments
    try:
        df = pd.read_csv("data/all_reviews.csv")
        sentiments = ["negative"]
        if process_neutral:
            sentiments.append("neutral")
        comments = df.loc[df["sentiment"].isin(sentiments), ["text"]].copy()
        comments["text"] = comments["text"].astype(str)
        comment_list = comments["text"].tolist()[:10]
    except FileNotFoundError:
        print("Error: data/all_reviews.csv not found. Exiting.")
        return

    # Process comments
    retriever = Retriever()
    recommender = Recommender(model_id=config["models"]["recommender"], retriever=retriever)
    recommendations = recommender.generate_batch(comment_list)

    # Update database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for comment, rec in zip(comment_list, recommendations):
        print(f"Comment: {comment}\nRecommendation: {rec}\n")
        cursor.execute("UPDATE feedback SET recommendations = ? WHERE comment = ?", (rec, comment))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()