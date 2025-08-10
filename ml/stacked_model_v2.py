# ml/stacked_model_v2.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

# === Пути ===
DATA_15M_PATH = "./data/processed/btc_usdt_15m.parquet"
DATA_1H_PATH = "./data/processed/btc_usdt_1h.parquet"
MODEL_SAVE_PATH = "./models/btc_long_stacked_v2_model_15m.pkl"
LSTM_MODEL_PATH = "./models/lstm_model_v2_15m.keras"
LOG_FILE = "./logs/model_training_log.csv"

# === Параметры ===
SEQUENCE_LENGTH = 90      # 60 свечей × 15m = 15 часов
MAX_HISTORY = 20000       # Максимум 20k свечей
HOLD_CANDLES = 4          # Держим 4 свечи = 60 минут (1 час)
TEST_SIZE = 72            # ~12 часов на валидацию (48 × 15m)
COMMISSION = 0.0008       # 0.08% комиссия

print("🔍 Обучение стекинг-модели: LSTM + XGBoost → LightGBM (meta) на 15m")

# === 1. Загружаем 15m данные ===
if not os.path.exists(DATA_15M_PATH):
    raise FileNotFoundError(f"Файл не найден: {DATA_15M_PATH}")

df = pd.read_parquet(DATA_15M_PATH)
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"✅ Загружено {len(df)} свечей (15m)")

# Ограничиваем историю
if len(df) > MAX_HISTORY:
    df = df.iloc[-MAX_HISTORY:].copy()
    print(f"📉 Ограничено до последних {MAX_HISTORY} свечей")

# === 2. Загружаем 1h данные для фильтра по тренду ===
if os.path.exists(DATA_1H_PATH):
    df_1h = pd.read_parquet(DATA_1H_PATH)
    df_1h = df_1h.sort_values("timestamp").reset_index(drop=True)
    # Привязываем последнюю 1h свечу к 15m
    last_1h_close = df_1h['close'].iloc[-1]
    df['trend_1h'] = (df['close'] > df['close'].rolling(50).mean()).astype(int)
    print("✅ Данные 1h загружены, добавлен фильтр по тренду")
else:
    print("⚠️ Файл 1h не найден. Продолжаем без фильтра.")
    df['trend_1h'] = 1  # По умолчанию разрешаем все сделки

# === 3. Фичи ===
def add_features(df):
    df = df.copy()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['volatility'] = df['high'] - df['low']
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['volume_ma'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    return df

df = add_features(df)

# Фичи для XGBoost
features = [
    'close', 'volume', 'sma_20', 'sma_50', 'volatility',
    'momentum', 'volume_ratio', 'trend_1h'
]

# === 4. Целевая переменная: закрытие выше SMA-20 через 4 свечи ===
df['future_close'] = df['close'].shift(-HOLD_CANDLES)
df['target'] = (df['future_close'] > df['sma_20']).astype(int)

# Удаляем NaN
df.dropna(subset=features + ['target'], inplace=True)
df = df.reset_index(drop=True)

# === 5. Подготовка данных для LSTM ===
def create_sequences(data, seq_len, features, target_col):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[features].iloc[i-seq_len:i].values)
        y.append(data[target_col].iloc[i])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(df, SEQUENCE_LENGTH, features, 'target')
X_flat = df[features].values[SEQUENCE_LENGTH:]
y_flat = df['target'].values[SEQUENCE_LENGTH:]

# Разделение
split_idx = len(X_seq) - TEST_SIZE
X_train_seq, X_val_seq = X_seq[:split_idx], X_seq[split_idx:]
X_train_flat, X_val_flat = X_flat[:split_idx], X_flat[split_idx:]
y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

print(f"📊 Тренировка: {len(X_train_seq)} | Валидация: {len(X_val_seq)}")

# === 6. LSTM модель (1-й уровень) ===
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

lstm_model = create_lstm_model((SEQUENCE_LENGTH, len(features)))
print("🧠 Обучение LSTM...")
lstm_model.fit(
    X_train_seq, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val_seq, y_val),
    verbose=1
)

# Прогнозы LSTM
lstm_train_proba = lstm_model.predict(X_train_seq).flatten()
lstm_val_proba = lstm_model.predict(X_val_seq).flatten()

# === 7. XGBoost модель (1-й уровень) ===
xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train_flat, y_train)

# Прогнозы XGBoost
xgb_train_proba = xgb_model.predict_proba(X_train_flat)[:, 1]
xgb_val_proba = xgb_model.predict_proba(X_val_flat)[:, 1]

# === 8. Стекинг: LightGBM как мета-модель (2-й уровень) ===
X_train_stack = np.column_stack([lstm_train_proba, xgb_train_proba])
X_val_stack = np.column_stack([lstm_val_proba, xgb_val_proba])

# Мета-модель — LightGBM
meta_model = LGBMClassifier(
    n_estimators=50,
    random_state=42,
    objective='binary',
    metric='binary_logloss'
)
meta_model.fit(X_train_stack, y_train)

# Финальный прогноз
y_proba_stacked = meta_model.predict_proba(X_val_stack)[:, 1]
y_pred_stacked = (y_proba_stacked > 0.6).astype(int)  # Повышенный порог

# === 9. Метрики ===
acc = accuracy_score(y_val, y_pred_stacked)
prec = precision_score(y_val, y_pred_stacked, zero_division=0)
rec = recall_score(y_val, y_pred_stacked, zero_division=0)
auc = roc_auc_score(y_val, y_proba_stacked)

print(f"\n📊 Результаты улучшенного стекинга (15m):")
print(f"  Accuracy: {acc:.3f}")
print(f"  Precision: {prec:.3f}")
print(f"  Recall: {rec:.3f}")
print(f"  AUC: {auc:.3f}")

# === 10. Сохранение моделей ===
os.makedirs("./models", exist_ok=True)

# Сохраняем мета-модель и базовые модели
joblib.dump({
    'xgb_model': xgb_model,
    'meta_model': meta_model,
    'sequence_length': SEQUENCE_LENGTH,
    'features': features
}, MODEL_SAVE_PATH)

# Сохраняем LSTM
lstm_model.save(LSTM_MODEL_PATH)

print("✅ Улучшенная стекинг-модель (15m) сохранена")

# === 11. Лог обучения ===
os.makedirs("./logs", exist_ok=True)
log_entry = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "model": "stacked_lstm_xgb_lgbm_15m",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "auc": auc,
    "status": "valid" if auc > 0.65 else "invalid"
}
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
print("✅ Лог обучения сохранён")