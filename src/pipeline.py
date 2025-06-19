import pandas as pd
import sqlite3
from typing import Union, Optional, List
from src.data.scraper import scrape_data  # Placeholder for scraper module
from src.data.attachment_processor import process_attachment
from src.data.database import init_db
from src.data.preprocess import preprocess_data
from scripts.populate_db import populate_db
from src.models.sentiment_generator import generate_sentiments
from src.utils.config import load_config


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

        # Get and preprocess input data
        preprocessed_df = None
        if use_scraper:
            print("Running scraper...")
            df = scrape_data()  # Assumes scrape_data() returns a DataFrame
            if df is None or df.empty:
                raise ValueError("Scraper returned no data")
            print(f"Scraped {len(df)} comments.")
            print("Preprocessing data...")
            preprocessed_df = preprocess_data(input_data=df)
        elif attachment is not None:
            print("Processing user-uploaded attachment...")
            if isinstance(attachment, pd.DataFrame):
                df = attachment
            else:
                df = process_attachment(attachment)
            if df.empty:
                raise ValueError("Attachment processing returned no data")
            print(f"Processed {len(df)} comments from attachment.")
            print("Preprocessing data...")
            preprocessed_df = preprocess_data(input_data=df)
        elif input_path:
            print(f"Using input CSV: {input_path}")
            print("Preprocessing data...")
            preprocessed_df = preprocess_data(input_path)
        else:
            print(f"Using default input CSV: {default_input_path}")
            print("Preprocessing data...")
            preprocessed_df = preprocess_data(default_input_path)

        if preprocessed_df.empty:
            print("No preprocessed data to insert. Exiting.")
            return
        print(f"Preprocessed {len(preprocessed_df)} comments.")

        # Insert comments into database
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
                    updated_count = len(sentiments_df)
                    print(f"Updated {updated_count} comments with sentiments.")

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
                    updated_count = len(aspects_df)
                    print(f"Updated {updated_count} comments with aspects.")

        print("Pipeline completed successfully.")

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")


if __name__ == "__main__":
    run_pipeline()