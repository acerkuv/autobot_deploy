# ml/stacked_model_v2_short_retrain.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import joblib
import os

# === Пути ===
DATA_15M_PATH = "./data/processed/btc_usdt_15m.parquet"
DATA_1H_PATH = "./data/processed/btc_usdt_1h.parquet"
DATA_4H_PATH = "./data/processed/btc_usdt_4h.parquet"
LOG_FILE = "./logs/model_training_log.csv"

# === Пути для сохранения экстракторов ===
LSTM_EXTRACTOR_PATH = "./models/lstm_extractor_v2_short.keras"
XGB_EXTRACTOR_PATH = "./models/xgb_extractor_v2_short.pkl"
SCALER_PATH = "./models/scaler_15m_short.pkl"
META_MODEL_PATH = "./models/btc_short_stacked_v2_model_15m_valid.pkl"

os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

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

# === Параметры (должны совпадать с LONG!) ===
SEQUENCE_LENGTH = 90      # ✅ Лучший рабочий сетап
MAX_HISTORY = 20000       # ✅ Ограничение данных
HOLD_CANDLES = 4          # Удержание: 60 минут
TEST_SIZE = 72            # Валидация: 3 дня
TARGET_THRESHOLD = 0.997  # SHORT: цена упадёт ниже на 0.3%

print("🔍 Обучение SHORT-стекинг модели: LSTM + XGBoost → LightGBM (meta) на 15m")

# === 1. Загружаем 15m данные ===
if not os.path.exists(DATA_15M_PATH):
    raise FileNotFoundError(f"Файл не найден: {DATA_15M_PATH}")

df = pd.read_parquet(DATA_15M_PATH)
df = df.sort_values("timestamp").reset_index(drop=True)
df = add_features(df)
print(f"✅ Загружено {len(df)} свечей (15m)")

if len(df) > MAX_HISTORY:
    df = df.iloc[-MAX_HISTORY:].copy()
    print(f"📉 Ограничено до последних {MAX_HISTORY} свечей")

# === 2. Загружаем 1h и 4h данные ===
if not os.path.exists(DATA_1H_PATH):
    raise FileNotFoundError(f"Файл не найден: {DATA_1H_PATH}")
if not os.path.exists(DATA_4H_PATH):
    raise FileNotFoundError(f"Файл не найден: {DATA_4H_PATH}")

df_1h = pd.read_parquet(DATA_1H_PATH).sort_values("timestamp").reset_index(drop=True)
df_4h = pd.read_parquet(DATA_4H_PATH).sort_values("timestamp").reset_index(drop=True)

df_1h = add_features(df_1h)
df_4h = add_features(df_4h)

# === 3. Мультитаймфреймовые фичи (для SHORT) ===
df['trend_1h'] = df['timestamp'].map(
    df_1h.set_index('timestamp')['close'] < df_1h.set_index('timestamp')['sma_50']  # медвежий тренд
)
df['volume_spike_4h'] = df['timestamp'].map(
    df_4h.set_index('timestamp')['volume'] > df_4h.set_index('timestamp')['volume'].rolling(10).mean() * 1.5
)

# ✅ Безопасное заполнение
df['trend_1h'] = pd.to_numeric(df['trend_1h'], errors='coerce').ffill().bfill().astype(int)
df['volume_spike_4h'] = pd.to_numeric(df['volume_spike_4h'], errors='coerce').ffill().bfill().astype(int)

# === 4. Фичи и таргет ===
features = [
    'close', 'volume', 'sma_20', 'sma_50', 'volatility',
    'momentum', 'volume_ma', 'volume_ratio', 'trend_1h', 'volume_spike_4h'
]

# SHORT: цена упадёт ниже на 0.3%
df['future_close'] = df['close'].shift(-HOLD_CANDLES)
df['target'] = (
    (df['future_close'] < df['close'] * TARGET_THRESHOLD) &
    (df['trend_1h'] == 1)
).astype(int)

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
scaler = StandardScaler()
X_train_seq_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
X_train_seq_scaled = scaler.fit_transform(X_train_seq_flat).reshape(X_train_seq.shape)

X_val_seq_flat = X_val_seq.reshape(-1, X_val_seq.shape[-1])
X_val_seq_scaled = scaler.transform(X_val_seq_flat).reshape(X_val_seq.shape)

# Сохраняем скалер
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Скалер сохранён: {SCALER_PATH}")

# === 7. LSTM (лучшая архитектура) ===
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
lstm_model.fit(
    X_train_seq_scaled, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val_seq_scaled, y_val),
    verbose=1
)

# === 8. Сохранение LSTM-эмбеддера (предпоследний слой) ===
intermediate_layer_model = Model(
    inputs=lstm_model.input,
    outputs=lstm_model.layers[-2].output
)
intermediate_layer_model.save(LSTM_EXTRACTOR_PATH)
print(f"✅ LSTM-эмбеддер сохранён: {LSTM_EXTRACTOR_PATH}")

# === 9. XGBoost ===
xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train_flat, y_train)

# Сохраняем XGBoost как экстрактор
joblib.dump(xgb_model, XGB_EXTRACTOR_PATH)
print(f"✅ XGBoost-экстрактор сохранён: {XGB_EXTRACTOR_PATH}")

# === 10. Прогнозы для стекинга ===
lstm_train_proba = lstm_model.predict(X_train_seq_scaled).flatten()
lstm_val_proba = lstm_model.predict(X_val_seq_scaled).flatten()
xgb_train_proba = xgb_model.predict_proba(X_train_flat)[:, 1]
xgb_val_proba = xgb_model.predict_proba(X_val_flat)[:, 1]

X_train_stack = np.column_stack([lstm_train_proba, xgb_train_proba])
X_val_stack = np.column_stack([lstm_val_proba, xgb_val_proba])

# === 11. Стекинг: LightGBM с балансировкой ===
sample_weights = compute_sample_weight("balanced", y_train)
meta_model = LGBMClassifier(n_estimators=50, random_state=42)
meta_model.fit(X_train_stack, y_train, sample_weight=sample_weights)

# === 12. Финальный прогноз (на валидации) ===
y_proba_stacked = meta_model.predict_proba(X_val_stack)[:, 1]
y_pred_stacked = (y_proba_stacked > 0.6).astype(int)

# === 13. Метрики ===
acc = accuracy_score(y_val, y_pred_stacked)
prec = precision_score(y_val, y_pred_stacked, zero_division=0)
rec = recall_score(y_val, y_pred_stacked, zero_division=0)
auc = roc_auc_score(y_val, y_proba_stacked)

print(f"\n📊 Результаты (SHORT): Accuracy={acc:.3f}, Precision={prec:.3f}, AUC={auc:.3f}")

# === 14. Сохранение мета-модели ===
joblib.dump({
    'meta_model': meta_model,
    'sequence_length': SEQUENCE_LENGTH,
    'features': features
}, META_MODEL_PATH)
print(f"✅ Мета-модель сохранена: {META_MODEL_PATH}")

# === 15. Лог обучения ===
log_entry = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "model": "btc_short_stacked_v2_retrain",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "auc": auc,
    "status": "valid" if auc > 0.65 else "invalid"
}
log_df = pd.DataFrame([log_entry])
log_df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
print("✅ Лог обучения сохранён")
print("✅ Все компоненты SHORT-модели сохранены. Готово к дообучению!")