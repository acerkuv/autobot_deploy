# ml/stacked_model_v2_short.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
import joblib
import os

# === Пути ===
DATA_15M_PATH = "./data/processed/btc_usdt_15m.parquet"
DATA_1H_PATH = "./data/processed/btc_usdt_1h.parquet"
DATA_4H_PATH = "./data/processed/btc_usdt_4h.parquet"
MODEL_SAVE_PATH = "./models/btc_short_stacked_v2_model_15m_valid.pkl"
LSTM_MODEL_PATH = "./models/lstm_model_v2_15m_short_valid.keras"
LOG_FILE = "./logs/model_training_log.csv"

# === Функция добавления фичей ===
def add_features(df):
    df = df.copy()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['volatility'] = df['high'] - df['low']
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['volume_ma'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    return df

# === Параметры ===
SEQUENCE_LENGTH = 120
MAX_HISTORY = 30000
HOLD_CANDLES = 4
TEST_SIZE = 72
TARGET_THRESHOLD = 0.997  # SHORT: цена упадёт ниже на 0.3%

print("🔍 Обучение SHORT-модели: LSTM + XGBoost → LightGBM (meta)")

# === 1. Загружаем 15m данные ===
if not os.path.exists(DATA_15M_PATH):
    raise FileNotFoundError(f"Файл не найден: {DATA_15M_PATH}")

df = pd.read_parquet(DATA_15M_PATH).sort_values("timestamp").reset_index(drop=True)
df = add_features(df)
print(f"✅ Загружено {len(df)} свечей (15m)")

if len(df) > MAX_HISTORY:
    df = df.iloc[-MAX_HISTORY:].copy()

# === 2. Загружаем 1h и 4h данные ===
df_1h = pd.read_parquet(DATA_1H_PATH).sort_values("timestamp").reset_index(drop=True)
df_4h = pd.read_parquet(DATA_4H_PATH).sort_values("timestamp").reset_index(drop=True)

df_1h = add_features(df_1h)
df_4h = add_features(df_4h)

# === 3. Мультитаймфреймовые фичи ===
df['trend_1h'] = df['timestamp'].map(df_1h.set_index('timestamp')['close'] < df_1h.set_index('timestamp')['sma_50'])  # Обратно: медвежий тренд
df['volume_spike_4h'] = df['timestamp'].map(
    df_4h.set_index('timestamp')['volume'] > df_4h.set_index('timestamp')['volume'].rolling(10).mean() * 1.5
)

df['trend_1h'] = pd.to_numeric(df['trend_1h'], errors='coerce').ffill().bfill().astype(int)
df['volume_spike_4h'] = pd.to_numeric(df['volume_spike_4h'], errors='coerce').ffill().bfill().astype(int)

# === 4. Фичи и таргет (SHORT) ===
features = [
    'close', 'volume', 'sma_20', 'sma_50', 'volatility',
    'momentum', 'volume_ratio', 'trend_1h', 'volume_spike_4h'
]

# Целевая переменная: цена упадёт ниже на 0.3%
df['future_close'] = df['close'].shift(-HOLD_CANDLES)
df['target'] = (df['future_close'] < df['close'] * TARGET_THRESHOLD).astype(int)

df.dropna(subset=features + ['target'], inplace=True)
df = df.reset_index(drop=True)

# === 5. Подготовка данных ===
def create_sequences(data, seq_len, features, target_col):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[features].iloc[i-seq_len:i].values)
        y.append(data[target_col].iloc[i])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(df, SEQUENCE_LENGTH, features, 'target')
X_flat = df[features].values[SEQUENCE_LENGTH:]
y_flat = df['target'].values[SEQUENCE_LENGTH:]

split_idx = len(X_seq) - TEST_SIZE
X_train_seq, X_val_seq = X_seq[:split_idx], X_seq[split_idx:]
X_train_flat, X_val_flat = X_flat[:split_idx], X_flat[split_idx:]
y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

# === 6. Нормализация для LSTM ===
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_seq_scaled = scaler.fit_transform(
    X_train_seq.reshape(-1, X_train_seq.shape[-1])
).reshape(X_train_seq.shape)
X_val_seq_scaled = scaler.transform(
    X_val_seq.reshape(-1, X_val_seq.shape[-1])
).reshape(X_val_seq.shape)

joblib.dump(scaler, "./models/scaler_15m_short.pkl")

# === 7. LSTM ===
def create_lstm_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=False, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

lstm_model = create_lstm_model((SEQUENCE_LENGTH, len(features)))

# === 8. Обучение LSTM ===
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lstm_model.fit(X_train_seq_scaled, y_train, epochs=15, batch_size=32, validation_data=(X_val_seq_scaled, y_val), callbacks=[early_stop], verbose=1)

# === 9. XGBoost ===
xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train_flat, y_train)

# === 10. Стекинг: LightGBM ===
lstm_train_proba = lstm_model.predict(X_train_seq_scaled).flatten()
lstm_val_proba = lstm_model.predict(X_val_seq_scaled).flatten()
xgb_train_proba = xgb_model.predict_proba(X_train_flat)[:, 1]
xgb_val_proba = xgb_model.predict_proba(X_val_flat)[:, 1]

X_train_stack = np.column_stack([lstm_train_proba, xgb_train_proba])
X_val_stack = np.column_stack([lstm_val_proba, xgb_val_proba])

meta_model = LGBMClassifier(n_estimators=50, random_state=42)
meta_model.fit(X_train_stack, y_train)

y_proba_stacked = meta_model.predict_proba(X_val_stack)[:, 1]
y_pred_stacked = (y_proba_stacked > 0.6).astype(int)

# === 11. Метрики ===
acc = accuracy_score(y_val, y_pred_stacked)
prec = precision_score(y_val, y_pred_stacked, zero_division=0)
rec = recall_score(y_val, y_pred_stacked, zero_division=0)
auc = roc_auc_score(y_val, y_proba_stacked)

print(f"\n📊 Результаты SHORT-модели: Accuracy={acc:.3f}, Precision={prec:.3f}, AUC={auc:.3f}")
print("✅ Модель SHORT обучена")

# === 12. Сохранение ===
os.makedirs("./models", exist_ok=True)
joblib.dump({
    'xgb_model': xgb_model,
    'meta_model': meta_model,
    'sequence_length': SEQUENCE_LENGTH,
    'features': features
}, MODEL_SAVE_PATH)
lstm_model.save(LSTM_MODEL_PATH)

# === 13. Лог ===
log_entry = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "model": "btc_short_stacked_v2",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "auc": auc,
    "status": "valid" if auc > 0.65 else "invalid"
}
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
print("✅ SHORT-модель сохранена")