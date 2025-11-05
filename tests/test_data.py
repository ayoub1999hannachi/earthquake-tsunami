
import os
from src.data_preprocessing import load_data, preprocess

def test_load_and_preprocess():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'earthquake_data_tsunami.csv')
    path = os.path.abspath(path)
    df = load_data(path)
    df2 = preprocess(df)
    assert 'tsunami' in df2.columns
    assert df2.isnull().sum().sum() == 0
