import pandas as pd
import sqlite3
from typing import Union, Optional, List
from src.data.scraper import scrape_data  # Placeholder for scraper module
from src.data.attachment_processor import process_attachment
from src.data.database import init_db
from src.data.preprocess import preprocess_data
from scripts.populate_db import populate_db
from src.utils.config import load_config


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame has required columns and is not empty.
    Args:
        df: DataFrame to validate.
    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    required_columns = {'text', 'sentiment'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")


def generate_sentiments(comments: List[str]) -> List[str]:
    """
    Placeholder: Generate sentiments for comments.
    Args:
        comments: List of comment texts.
    Returns:
        List of sentiments (e.g., 'positive', 'negative', 'neutral').
    """
    # TODO: Implement sentiment generation logic (e.g., using a model)
    print("Generating sentiments (placeholder)...")
    return ['neutral'] * len(comments)  # Dummy output


def generate_aspects(comments: List[str]) -> List[Optional[str]]:
    """
    Placeholder: Generate aspects for comments.
    Args:
        comments: List of comment texts.
    Returns:
        List of aspect strings or None.
    """
    # TODO: Implement aspect generation logic (e.g., using a model)
    print("Generating aspects (placeholder)...")
    return [None] * len(comments)  # Dummy output


def run_pipeline(
    use_scraper: bool = False,
    attachment: Optional[Union[str, pd.DataFrame]] = None,
    input_path: Optional[str] = None
) -> None:
    """
    Orchestrate the data processing pipeline: scrape or process attachments, preprocess, insert comments,
    then generate and update sentiments and aspects separately.
    Args:
        use_scraper: If True, run scraper to fetch data.
        attachment: User-uploaded CSV file (path or DataFrame).
        input_path: Path to input CSV if not using scraper or attachment.
    """
    try:
        config = load_config()
        db_path = config["database"]["path"]
        default_input_path = config["processing"].get("input_path", "data/all_reviews.csv")
        use_sentiment_generation = config["processing"].get("use_sentiment_generation", False)
        use_aspect_generation = config["processing"].get("use_aspect_generation", False)

        # Initialize database
        init_db(db_path)
        print("Database initialized.")

        # Get input data
        df = None
        if use_scraper:
            print("Running scraper...")
            df = scrape_data()  # Assumes scrape_data() returns a DataFrame
            if df is None:
                raise ValueError("Scraper returned no data")
            validate_dataframe(df)
            print(f"Scraped {len(df)} comments.")
        elif attachment is not None:
            print("Processing user-uploaded attachment...")
            if isinstance(attachment, pd.DataFrame):
                df = attachment
                validate_dataframe(df)
            else:
                df = process_attachment(attachment)
                validate_dataframe(df)
            print(f"Processed {len(df)} comments from attachment.")
        elif input_path:
            print(f"Using input CSV: {input_path}")
            df = preprocess_data(input_path)  # Let preprocess_data handle CSV loading
            validate_dataframe(df)
            print(f"Loaded {len(df)} comments from CSV.")
        else:
            print(f"Using default input CSV: {default_input_path}")
            df = preprocess_data(default_input_path)
            validate_dataframe(df)
            print(f"Loaded {len(df)} comments from default CSV.")

        # Preprocess and insert comments
        print("Preprocessing data...")
        preprocessed_df = preprocess_data(input_data=df)
        if preprocessed_df.empty:
            print("No preprocessed data to insert. Exiting.")
            return
        print(f"Preprocessed {len(preprocessed_df)} comments.")

        print("Inserting comments into database...")
        populate_db(input_data=preprocessed_df, generate_recommendations=False)
        print("Comments inserted successfully.")

        # Generate and update sentiments
        if use_sentiment_generation:
            print("Generating sentiments...")
            with sqlite3.connect(db_path) as conn:
                sentiments_df = pd.read_sql_query("""
                    SELECT comment, comment_id
                    FROM feedback
                    WHERE tone IS NULL
                """, conn)
                if sentiments_df.empty:
                    print("No comments need sentiment generation.")
                else:
                    sentiments = generate_sentiments(sentiments_df['comment'].tolist())
                    cursor = conn.cursor()
                    cursor.executemany("""
                        UPDATE feedback
                        SET tone = ?
                        WHERE comment_id = ?
                    """, list(zip(sentiments, sentiments_df['comment_id'])))
                    conn.commit()
                    print(f"Updated {len(sentiments_df)} comments with sentiments.")

        # Generate and update aspects
        if use_aspect_generation:
            print("Generating aspects...")
            with sqlite3.connect(db_path) as conn:
                aspects_df = pd.read_sql_query("""
                    SELECT comment, comment_id
                    FROM feedback
                    WHERE aspect IS NULL
                """, conn)
                if aspects_df.empty:
                    print("No comments need aspect generation.")
                else:
                    aspects = generate_aspects(aspects_df['comment'].tolist())
                    cursor = conn.cursor()
                    cursor.executemany("""
                        UPDATE feedback
                        SET aspect = ?
                        WHERE comment_id = ?
                    """, list(zip(aspects, aspects_df['comment_id'])))
                    conn.commit()
                    print(f"Updated {len(aspects_df)} comments with aspects.")

        print("Pipeline completed successfully.")

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")


if __name__ == "__main__":
    run_pipeline()