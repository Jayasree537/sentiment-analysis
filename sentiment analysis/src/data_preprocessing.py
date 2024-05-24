import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    data_path = r'data/raw/sample_data.csv'
    data = pd.read_csv(data_path)
    return data['text'], data['sentiment']

def split_data(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


