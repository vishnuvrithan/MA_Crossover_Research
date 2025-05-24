"""
ml.py: Hooks for ML model training and feature generation
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_feature_matrix(data: pd.DataFrame):
    X = pd.DataFrame({
        'MA_5': data['Close'].rolling(5).mean(),
        'MA_10': data['Close'].rolling(10).mean(),
        'MA_ratio': data['Close'].rolling(5).mean() / data['Close'].rolling(10).mean(),
    })
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    X = X.fillna(0)
    return X, y


def train_hybrid_model(data: pd.DataFrame):
    X, y = create_feature_matrix(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc
