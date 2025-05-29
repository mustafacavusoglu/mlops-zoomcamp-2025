import pandas as pd

def read_data(file_path):
    """
    Reads a parquet file and returns a DataFrame."""
    
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None
