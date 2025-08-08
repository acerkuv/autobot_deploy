# ml/train.py
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import joblib
import os

# Загружаем данные
df_3m = pd.read_parquet("./data/processed/btc_usdt_3m.parquet")
df_1h = pd.read_parquet("./data/processed/btc_usdt_1h.parquet")

print(" Обучение ансамбля XGBoost + LightGBM...")

# Работаем с 3m данными
df = df_3m.copy()

# Целевая переменная: будет ли рост через 12 свечей (36 минут)?
df['future_close'] = df['close'].shift(-12)
df['target'] = (df['future_close'] > df['close'] * 1.005).astype(int)  # +0.5%

# === Расчёт фичей ===
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()
df['volatility'] = df['high'] - df['low']
df['momentum'] = df['close'] - df['close'].shift(5)
df['volume_ma'] = df['volume'].rolling(10).mean()
df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)  # Защита от деления на 0

# Удаляем строки с NaN
df.dropna(inplace=True)

# Выбираем фичи
features = [
    'close', 'volume', 'sma_20', 'sma_50', 'volatility',
    'momentum', 'volume_ratio'
]
X = df[features]
y = df['target']

# Разделение на обучение и валидацию (последние 12 часов)
split_idx = len(X) - 240  # ~12 часов
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Модели
xgb = XGBClassifier(n_estimators=100, random_state=42)
lgb = LGBMClassifier(
    n_estimators=100,
    random_state=42,
    scale_pos_weight=20750 / 549  # ≈37.8
)
model = LGBMClassifier(
    n_estimators=100,
    random_state=42,
    scale_pos_weight=40,
    objective='binary',
    metric='binary_logloss'
)

# Обучение
model.fit(X_train, y_train)

# Валидация
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)

print(f"📊 Валидация: Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")

# Сохранение
os.makedirs("./models", exist_ok=True)
joblib.dump(model, "./models/btc_long_model.pkl")
print("✅ Модель сохранена")