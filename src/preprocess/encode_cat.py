import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_categorical(data, columns):
    """
    One-hot encode categorical columns.
    """
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(data[columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns))
    return pd.concat([data.drop(columns, axis=1), encoded_df], axis=1)