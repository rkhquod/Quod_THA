"""
Contains functions to:
 - Load CSV data
 - Clean and preprocess the dataset
"""

import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data from a file_path.
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning:
      - Rename columns to have consistent naming (if needed).
      - Convert 'date' column to datetime.
      - Drop invalid dates if necessary.
      - Sort by date.
      - Drop duplicates if any.
    """
    df.rename(columns={
        'Customer ID': 'customer_id',
        'Product': 'product_id',
        'Time stamp': 'date'
    }, inplace=True, errors='ignore') 

    # Convert to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Drop rows with invalid/missing dates
        df.dropna(subset=['date'], inplace=True)
        # Sort by date
        df.sort_values(by='date', inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
