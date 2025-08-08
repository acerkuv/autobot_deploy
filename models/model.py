# models/model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(df_3m, df_1h):
    """
    Обучает модель на данных 3m и 1h.
    """
    print("🚀 Запуск обучения модели на 3m и 1h данных...")

    # Работаем с 3m данными
    df = df_3m.copy()

    # Целевая переменная: будет ли рост через 12 свечей (36 минут)?
    df['future_close'] = df['close'].shift(-12)
    df['target'] = (df['future_close'] > df['close'] * 1.005).astype(int)  # +0.5%

    # Фичи
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['volatility'] = df['high'] - df['low']
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['volume_ma'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Выбираем фичи
    feature_columns = [
        'close', 'volume', 'sma_20', 'sma_50', 'volatility',
        'momentum', 'volume_ratio'
    ]

    # Удаляем строки с NaN
    df.dropna(subset=feature_columns + ['target'], inplace=True)

    X = df[feature_columns]
    y = df['target']

    if len(y) == 0:
        print("❌ Нет данных для обучения после очистки")
        return None

    # Обучаем модель
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Сохраняем модель
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/btc_long_model.pkl")
    print(f"✅ Модель обучена на {len(X)} свечах и сохранена в models/btc_long_model.pkl")

    return model