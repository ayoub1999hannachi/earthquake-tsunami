"""Utilities for loading and preprocessing the earthquake dataset."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'raw', 'earthquake_data_tsunami.csv')


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load dataset CSV from path (default DATA_PATH).
    Raises FileNotFoundError if path does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal preprocessing:
    - drop exact duplicates
    - fill numeric NA with mean
    - ensure tsunami is int
    """
    df = df.copy()
    # drop exact duplicates
    df = df.drop_duplicates()
    # fill na with mean for numeric cols
    for c in df.select_dtypes(include='number').columns:
        df[c] = df[c].fillna(df[c].mean())
    # ensure types
    if 'tsunami' in df.columns:
        df['tsunami'] = df['tsunami'].astype(int)
    return df


def split_features(df: pd.DataFrame, target: str = 'tsunami'):
    """
    Split into X, y and return train/test
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)
