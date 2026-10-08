import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame, raising FileNotFoundError if it doesn't exist."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)
