import pandas as pd
import sqlite3
import json
from typing import Union, Optional
from src.data.scraper import scrape_data  # Placeholder for scraper module
from src.data.attachment_processor import process_attachment
from src.data.database import init_db
from src.data.preprocess import preprocess_data
from scripts.populate_db import populate_db
from src.models.sentiment_generator import generate_sentiments
from src.models.aspect_generator import generate_aspects

from src.utils.config import load_config


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
        if config["processing"]["allow_competitors"]:
            default_input_path = config["processing"]["competitor_path"]
        else:
            default_input_path = config["processing"]["input_path"]

        use_sentiment_generation = config["processing"]["use_sentiment_generation"]
        use_aspect_generation = config["processing"]["use_aspect_generation"]

        # Initialize database
        init_db(db_path)
        print("Database initialized.")

        # Get and preprocess input data
        df = None  # initialized once

        # Get input data
        if use_scraper:
            print("Running scraper...")
            df = scrape_data()
            if df is None or df.empty:
                raise ValueError("Scraper returned no data")
        elif attachment is not None:
            print("Processing user-uploaded attachment...")
            df = attachment if isinstance(attachment, pd.DataFrame) else process_attachment(attachment)
            if df.empty:
                raise ValueError("Attachment processing returned no data")
        else:
            input_path = input_path or default_input_path
            print(f"Using input file: {input_path}")

        # Preprocess
        preprocessed_df = preprocess_data(input_data=df if df is not None else input_path)

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
                # Sentiment
                sentiments_df = pd.read_sql_query("""
                    SELECT comment, comment_id
                    FROM feedback
                    WHERE tone IS NULL OR tone = ''
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
                # Aspect
                aspects_df = pd.read_sql_query("""
                    SELECT comment, comment_id
                    FROM feedback
                    WHERE aspect IS NULL OR aspect = ''
                """, conn)

                if aspects_df.empty:
                    print("No comments need aspect generation.")
                else:
                    # Step 1: generate aspect lists
                    aspects_lists = generate_aspects(aspects_df['comment'].tolist())  # should return List[List[str]]

                    # Step 2: convert to JSON strings for DB storage
                    aspects_json = [json.dumps(aspect_list, ensure_ascii=False) for aspect_list in aspects_lists]

                    # Step 3: prepare update data (json_str, comment_id)
                    update_data = list(zip(aspects_json, aspects_df['comment_id']))

                    # Step 4: update database
                    cursor = conn.cursor()
                    cursor.executemany("""
                        UPDATE feedback
                        SET aspect = ?
                        WHERE comment_id = ?
                    """, update_data)

                    conn.commit()
                    updated_count = len(update_data)
                    print(f"Updated {updated_count} comments with aspects.")

        print("Pipeline completed successfully.")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Invalid data: {e}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    run_pipeline()