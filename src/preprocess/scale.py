import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
def scale_data(data, method='standard'):
    """
    Scale numerical data.
    method: 'standard' or 'minmax'
    """
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)