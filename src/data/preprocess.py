import pandas as pd
import hashlib
import sqlite3
import pickle
from typing import Optional, Set, Union
from sklearn.metrics.pairwise import cosine_similarity
from src.models.embedder import Embedder
from src.utils.config import load_config
import nltk


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


def remove_stopwords(text: str, stop_words: Set[str]) -> str:
    """
    Remove stopwords from text (optional).
    Args:
        text: Input text.
        stop_words: Set of stopwords to remove (not mutated).
    Returns:
        Text without stopwords.
    """
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


def preprocess_data(input_data: Union[str, pd.DataFrame] = "data/all_reviews.csv",
                    product_description: Optional[str] = None,
                    relevance_threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Preprocess reviews: remove duplicates (from CSV and database), optionally filter relevant feedback,
    optionally remove stopwords, generate embeddings, and return a DataFrame.
    Args:
        input_data: Path to input CSV or DataFrame (from scraper or attachment).
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
        stopwords_language = config["processing"].get("stopwords_language", "russian")
        db_path = config["database"]["path"]
        default_brand = config["processing"].get("default_brand", "PPR")

        if product_description is None:
            product_description = config["processing"]["product_description"]
        if relevance_threshold is None:
            relevance_threshold = config["processing"]["relevance_threshold"]

        embedder = Embedder()

        # Check and download NLTK stopwords once
        if remove_stopwords_flag:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
        stop_words = set(nltk.corpus.stopwords.words(stopwords_language)) if remove_stopwords_flag else set()

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

        # Load input data
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
            if not {'text', 'sentiment'}.issubset(df.columns):
                raise ValueError("DataFrame must contain 'text' and 'sentiment' columns")
        else:
            try:
                df = pd.read_csv(input_data, sep=';', encoding="utf-8", usecols=['text', 'sentiment'])
            except (UnicodeDecodeError, FileNotFoundError, ValueError) as e:
                try:
                    df = pd.read_csv(input_data, sep=';', encoding="utf-8-sig", usecols=['text', 'sentiment'])
                except (UnicodeDecodeError, FileNotFoundError, ValueError):
                    try:
                        df = pd.read_csv(input_data, sep=';', encoding="cp1252", usecols=['text', 'sentiment'])
                    except FileNotFoundError:
                        if not use_fallback:
                            raise FileNotFoundError(
                                f"{input_data} not found. Set 'use_fallback_data: true' in config.yaml to use sample data.")
                        print(f"Error: {input_data} not found. Using sample data.")
                        df = pd.DataFrame([
                            {"text": "Долго возвращают средства", "sentiment": "negative"},
                            {"text": "Приложение постоянно крашится", "sentiment": "negative"},
                            {"text": "Приложение крашится", "sentiment": "negative"},
                            {"text": "Поддержка работает окей", "sentiment": "neutral"},
                            {"text": "Не связано с продуктом", "sentiment": "negative"}
                        ])
                    except ValueError as ve:
                        raise ValueError(f"CSV must contain 'text' and 'sentiment' columns: {str(ve)}")

        # Generate comment IDs and rename columns for database
        df['comment_id'] = df['text'].apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())
        df = df.rename(columns={"text": "comment", "sentiment": "tone"})

        # Remove duplicates within the data
        csv_duplicates = df.duplicated(subset=['comment_id'], keep='first')
        if csv_duplicates.sum() > 0:
            print(f"Removed {csv_duplicates.sum()} duplicate comments within the data.")
            df = df[~csv_duplicates].copy()

        # Remove comments that already exist in the database
        db_duplicates = df['comment_id'].isin(existing_comment_ids)
        if db_duplicates.sum() > 0:
            print(f"Removed {db_duplicates.sum()} comments that already exist in the database.")
            df = df[~db_duplicates].copy()

        # Apply stopwords removal vectorized
        if remove_stopwords_flag:
            df['comment'] = df['comment'].str.split().apply(
                lambda words: " ".join(word for word in words if word.lower() not in stop_words))

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
            df = df[df['is_relevant']].drop(columns=['is_relevant']).copy()
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

        return df

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return pd.DataFrame()