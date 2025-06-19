import pandas as pd
import hashlib
import sqlite3
import pickle
from typing import Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
from src.models.embedder import Embedder
from src.utils.config import load_config
from src.data.database import init_db
import nltk
from nltk.corpus import stopwords


def get_feedback_relevance(comment: str, embedder: Embedder, product_emb: list, threshold: float = 0.5) -> bool:
    """
    Determine if feedback is relevant by comparing embeddings of comment and product description.
    Args:
        comment: Customer feedback text.
        embedder: Embedder instance for generating embeddings.
        product_emb: Precomputed embedding of the product description.
        threshold: Cosine similarity threshold for relevance (default: 0.5).
    Returns:
        True if relevant, False otherwise.
    """
    embedding = embedder.encode(comment)
    similarity = cosine_similarity([product_emb], [embedding])[0][0]
    return similarity >= threshold


def remove_stopwords(text: str, language: str = "russian") -> str:
    """
    Remove stopwords from text (optional).
    Args:
        text: Input text.
        language: Language for stopwords (e.g., 'russian', 'english').
    Returns:
        Text without stopwords.
    """
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    stop_words = set(stopwords.words(language))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


def preprocess_data(input_path: str = "data/all_reviews.csv", product_description: Optional[str] = None,
                    relevance_threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Preprocess reviews: remove duplicates (from CSV and database), optionally filter relevant feedback,
    optionally remove stopwords, generate embeddings, and insert directly into the database.
    Args:
        input_path: Path to input CSV (from scraper.py).
        product_description: Product description for relevance check.
        relevance_threshold: Cosine similarity threshold.
    Returns:
        Preprocessed DataFrame.
    """
    try:
        config = load_config()
        use_fallback = config["processing"]["use_fallback_data"]
        use_relevance_filter = config["processing"]["use_relevance_filter"]
        remove_stopwords_flag = config["processing"]["remove_stopwords"]
        stopwords_language = config["processing"]["stopwords_language"]
        db_path = config["database"]["path"]
        default_brand = config["processing"]["default_brand"]  # Default brand from config

        if product_description is None:
            product_description = config["processing"]["product_description"]
        if relevance_threshold is None:
            relevance_threshold = config["processing"]["relevance_threshold"]

        embedder = Embedder()

        # Initialize database
        init_db(db_path)

        # Load existing comment IDs from the database
        existing_comment_ids: Set[str] = set()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT comment_id FROM feedback")
                existing_comment_ids = {row[0] for row in cursor.fetchall()}
                print(f"Found {len(existing_comment_ids)} existing comments in the database.")
            except sqlite3.OperationalError:
                print("Database table 'feedback' not found. No existing comments to check for duplicates.")

        # Load input CSV, validate 'text' and 'sentiment' columns, or use fallback
        try:
            df = pd.read_csv(input_path, sep=';', encoding="cp1252")
            if not {'text', 'sentiment'}.issubset(df.columns):
                raise ValueError("CSV must contain 'text' and 'sentiment' columns")
        except FileNotFoundError:
            if not use_fallback:
                raise FileNotFoundError(
                    f"{input_path} not found. Set 'use_fallback_data: true' in config.yaml to use sample data.")
            print(f"Error: {input_path} not found. Using sample data.")
            df = pd.DataFrame([
                {"text": "Долго возвращают средства", "sentiment": "negative"},
                {"text": "Приложение постоянно крашится", "sentiment": "negative"},
                {"text": "Приложение крашится", "sentiment": "negative"},
                {"text": "Поддержка работает окей", "sentiment": "neutral"},
                {"text": "Не связано с продуктом", "sentiment": "negative"}
            ])

        # Generate comment IDs and rename columns for database
        df['comment_id'] = df['text'].apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())
        df = df.rename(columns={"text": "comment", "sentiment": "tone"})

        # Remove duplicates within the CSV
        csv_duplicates = df.duplicated(subset=['comment_id'], keep='first')
        if csv_duplicates.sum() > 0:
            print(f"Removed {csv_duplicates.sum()} duplicate comments within the CSV.")
            df = df[~csv_duplicates]

        # Remove comments that already exist in the database
        db_duplicates = df['comment_id'].isin(existing_comment_ids)
        if db_duplicates.sum() > 0:
            print(f"Removed {db_duplicates.sum()} comments that already exist in the database.")
            df = df[~db_duplicates]

        if remove_stopwords_flag:
            df['comment'] = df['comment'].apply(lambda x: remove_stopwords(x, language=stopwords_language))

        # Generate embeddings once for all comments
        df['embedding_vector'] = df['comment'].apply(embedder.encode)

        if use_relevance_filter:
            product_emb = embedder.encode(product_description)
            df['is_relevant'] = df['embedding_vector'].apply(
                lambda emb: cosine_similarity([product_emb], [emb])[0][0] >= relevance_threshold
            )
            irrelevant_count = (~df['is_relevant']).sum()
            if irrelevant_count > 0:
                print(f"Filtered out {irrelevant_count} irrelevant comments")
            df = df[df['is_relevant']].drop(columns=['is_relevant'])
        else:
            print("Relevance filtering disabled; keeping all non-duplicate comments")

        # Log total comments after filtering
        print(f"Total comments after filtering: {len(df)}")

        # Add columns required for the feedback table
        df['brand'] = default_brand
        df['aspect'] = None
        df['recommendations'] = None
        # Pickle embeddings for database insertion
        df['embedding'] = df['embedding_vector'].apply(pickle.dumps)
        df.drop(columns=['embedding_vector'], inplace=True)

        # Insert preprocessed data into the database
        # Note: For large datasets, consider batch inserts or df.to_sql with method='multi' for performance.
        # INSERT OR IGNORE relies on comment_id PRIMARY KEY to skip duplicates.
        if not df.empty:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR IGNORE INTO feedback (comment_id, brand, comment, tone, aspect, recommendations, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, df[['comment_id', 'brand', 'comment', 'tone', 'aspect', 'recommendations',
                         'embedding']].values.tolist())
                conn.commit()
                inserted_count = cursor.rowcount
                print(f"Inserted {inserted_count} preprocessed reviews into the database at {db_path}")
        else:
            print("No new preprocessed reviews to insert into the database.")

        return df

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return pd.DataFrame()


if __name__ == "__main__":
    preprocess_data()