import pandas as pd
from typing import Union, Optional
from io import StringIO


def process_attachment(file: Union[str, bytes, StringIO], encoding: Optional[str] = None) -> pd.DataFrame:
    """
    Process a user-uploaded CSV file into a standardized DataFrame.
    Args:
        file: File path, bytes, or StringIO object containing CSV data.
        encoding: File encoding (e.g., 'utf-8', 'utf-8-sig', 'cp1252'). Defaults to auto-detect.
    Returns:
        DataFrame with 'text' and 'sentiment' columns.
    Raises:
        ValueError: If CSV lacks required columns or is malformed.
        FileNotFoundError: If file path is invalid.
    """
    try:
        encodings = [encoding, 'utf-8', 'utf-8-sig', 'cp1252'] if encoding else ['utf-8', 'utf-8-sig', 'cp1252']
        df = None
        last_error = None

        for enc in encodings:
            try:
                if isinstance(file, str):
                    df = pd.read_csv(file, sep=';', usecols=['text', 'sentiment'], encoding=enc)
                elif isinstance(file, (bytes, StringIO)):
                    if isinstance(file, bytes):
                        file = StringIO(file.decode(enc))
                    df = pd.read_csv(file, sep=';', usecols=['text', 'sentiment'])
                break
            except (UnicodeDecodeError, ValueError, FileNotFoundError) as e:
                last_error = e
                continue

        if df is None:
            raise ValueError(f"Failed to read CSV with encodings {encodings}: {str(last_error)}")

        if not {'text', 'sentiment'}.issubset(df.columns):
            raise ValueError("CSV must contain 'text' and 'sentiment' columns")

        # Safely parse 'aspect' column if it exists (avoid float iteration error)
        if 'aspect' in df.columns:
            from ast import literal_eval

            def safe_eval(x):
                import pandas as pd
                if pd.isna(x):
                    return []
                if isinstance(x, str) and x.strip().startswith('['):
                    try:
                        return literal_eval(x)
                    except:
                        return []
                return []

            df['parsed_aspects'] = df['aspect'].apply(safe_eval)
        else:
            df['parsed_aspects'] = [[] for _ in range(len(df))]

        return df

    except Exception as e:
        print(f"Error processing attachment: {str(e)}")
        raise