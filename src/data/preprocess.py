import pandas as pd
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from src.models.embedder import Embedder
from src.utils.config import load_config
import nltk
from nltk.corpus import stopwords


def get_feedback_relevance(comment, product_description, embedder, threshold=0.5):
    """
    Determine if feedback is relevant by comparing embeddings of comment and product description.
    Args:
        comment (str): Customer feedback text.
        product_description (str): Description of the product/service.
        embedder (Embedder): Embedder instance for generating embeddings.
        threshold (float): Cosine similarity threshold for relevance (default: 0.5).
    Returns:
        bool: True if relevant, False otherwise.
    """
    embedding1 = embedder.encode(product_description)
    embedding2 = embedder.encode(comment)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity >= threshold


def remove_stopwords(text, language="russian"):
    """
    Remove stopwords from text (optional).
    Args:
        text (str): Input text.
        language (str): Language for stopwords (e.g., 'russian', 'english').
    Returns:
        str: Text without stopwords.
    """
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    stop_words = set(stopwords.words(language))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


def preprocess_data(input_path="data/all_reviews.csv", output_path="data/preprocessed_reviews.csv",
                    product_description=None, relevance_threshold=None):
    """
    Preprocess reviews: remove duplicates, optionally filter relevant feedback, optionally remove stopwords, optionally
    use fallback data.
    Args:
        input_path (str): Path to input CSV (from scraper.py).
        output_path (str): Path to save preprocessed CSV.
        product_description (str, optional): Product description for relevance check.
        relevance_threshold (float, optional): Cosine similarity threshold.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    config = load_config()
    use_fallback = config["processing"]["use_fallback_data"]
    use_relevance_filter = config["processing"]["use_relevance_filter"]
    remove_stopwords_flag = config["processing"]["remove_stopwords"]
    product_description = product_description or config["processing"]["product_description"]
    relevance_threshold = relevance_threshold or config["processing"]["relevance_threshold"]
    embedder = Embedder()

    # Load input CSV, validate 'text' and 'sentiment' columns, or use fallback
    # sample data if file is missing and use_fallback_data is true
    try:
        df = pd.read_csv(input_path, sep=';', encoding="cp1252")
        if not {'text', 'sentiment'}.issubset(df.columns):
            raise ValueError("CSV must contain 'text' and 'sentiment' columns")
    except FileNotFoundError:
        if not use_fallback:
            raise FileNotFoundError(f"{input_path} not found. Set 'use_fallback_data: true' in config.yaml to use \
             sample data.")
        print(f"Error: {input_path} not found. Using sample data.")
        df = pd.DataFrame([
            {"text": "Долго возвращают средства", "sentiment": "negative"},
            {"text": "Приложение постоянно крашится", "sentiment": "negative"},
            {"text": "Приложение крашится", "sentiment": "negative"},
            {"text": "Поддержка работает окей", "sentiment": "neutral"},
            {"text": "Не связано с продуктом", "sentiment": "negative"}
        ])

    df['comment_id'] = df['text'].apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())
    duplicates = df.duplicated(subset=['comment_id'], keep='first')
    if duplicates.sum() > 0:
        print(f"Removed {duplicates.sum()} duplicate comments")
        df = df[~duplicates]

    if remove_stopwords_flag:
        df['text'] = df['text'].apply(lambda x: remove_stopwords(x, language="russian"))

    if use_relevance_filter:
        df['is_relevant'] = df['text'].apply(
            lambda x: get_feedback_relevance(x, product_description, embedder, relevance_threshold)
        )
        irrelevant_count = (~df['is_relevant']).sum()
        if irrelevant_count > 0:
            print(f"Filtered out {irrelevant_count} irrelevant comments")
        df = df[df['is_relevant']].drop(columns=['is_relevant'])
    else:
        print("Relevance filtering disabled; keeping all non-duplicate comments")

    df.to_csv(output_path, index=False, sep=';', encoding="cp1252")
    print(f"Saved {len(df)} preprocessed reviews to {output_path}")

    return df


if __name__ == "__main__":
    preprocess_data()