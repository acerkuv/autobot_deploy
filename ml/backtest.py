# ml/backtest.py
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from ml.features import create_features

def backtest_model(model, df, window_days=30):
    df = df.tail(4 * 24 * window_days)
    df = create_features(df)
    if len(df) == 0:
        return None
    X = df[['return', 'sma_5', 'sma_20', 'spread', 'volume_ma', 'rsi']]
    y = df['target']
    if len(X) == 0 or X.isna().any().any():
        return None
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    return {"days": window_days, "accuracy": acc, "precision": prec}