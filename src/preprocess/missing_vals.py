import pandas as pd

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in the dataset.
    strategy: 'mean', 'median', 'drop', or 'zero'
    """
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'drop':
        return data.dropna()
    elif strategy == 'zero':
        return data.fillna(0)
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'drop', or 'zero'.")
