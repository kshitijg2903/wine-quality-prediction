import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def load_and_split_data(filepath):
    data = pd.read_csv(filepath)

    """
    given some new dataset: 
    Need to figure out how to script this so that 
    it figures out which column is the target variable
    """

    if 'Id' in data.columns:
        data = data.drop(columns=['Id'])
    X = data.drop(columns = ['quality'])
    y = data['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

