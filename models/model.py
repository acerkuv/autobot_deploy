# models/model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(df_3m, df_1h):
    """
    Обучает модель на мультифреймовых данных.
    """
    # Пример простой модели
    df = df_3m[['open', 'high', 'low', 'close', 'volume_btc']].copy()
    df['return'] = df['close'].pct_change(12)  # 12 свечей вперёд
    df['target'] = (df['return'] > 0.001).astype(int)
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['signal'] = (df['sma_20'] > df['sma_50']).astype(int)

    # Удаляем NaN
    df.dropna(inplace=True)

    X = df[['close', 'volume_btc', 'sma_20', 'sma_50', 'signal']]
    y = df['target']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model

# def train_model(df_3m, df_1h):
#     print("Модель заглушка: обучение имитируется")
#     return "fake_model_v1"