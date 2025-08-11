# ml/backtest_stacked_short.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# === Пути ===
MODEL_PATH = "./models/btc_short_stacked_v2_model_15m_valid.pkl"
LSTM_MODEL_PATH = "./models/lstm_model_v2_15m_short_valid.keras"
SCALER_PATH = "./models/scaler_15m_short.pkl"
DATA_15M_PATH = "./data/processed/btc_usdt_15m.parquet"
DATA_1H_PATH = "./data/processed/btc_usdt_1h.parquet"
DATA_4H_PATH = "./data/processed/btc_usdt_4h.parquet"
LOG_FILE = "./logs/backtest_stacked_short_log.csv"

# === Параметры ===
SEQUENCE_LENGTH = 120
HOLD_CANDLES = 4
COMMISSION = 0.0008
INITIAL_CAPITAL = 1000.0

print("🔍 Запуск бэкте́ста SHORT-модели")

# === 1. Загрузка данных ===
df_15m = pd.read_parquet(DATA_15M_PATH).sort_values("timestamp").reset_index(drop=True)
df_1h = pd.read_parquet(DATA_1H_PATH).sort_values("timestamp").reset_index(drop=True)
df_4h = pd.read_parquet(DATA_4H_PATH).sort_values("timestamp").reset_index(drop=True)

# === 2. Фичи ===
def add_features(df):
    df = df.copy()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['volatility'] = df['high'] - df['low']
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['volume_ma'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    return df

df_15m = add_features(df_15m)
df_1h = add_features(df_1h)
df_4h = add_features(df_4h)

# === 3. Мультитаймфреймовые фичи ===
df_15m['trend_1h'] = df_15m['timestamp'].map(df_1h.set_index('timestamp')['close'] < df_1h.set_index('timestamp')['sma_50'])
df_15m['volume_spike_4h'] = df_15m['timestamp'].map(
    df_4h.set_index('timestamp')['volume'] > df_4h.set_index('timestamp')['volume'].rolling(10).mean() * 1.5
)

df_15m['trend_1h'] = pd.to_numeric(df_15m['trend_1h'], errors='coerce').ffill().bfill().astype(int)
df_15m['volume_spike_4h'] = pd.to_numeric(df_15m['volume_spike_4h'], errors='coerce').ffill().bfill().astype(int)

features = [
    'close', 'volume', 'sma_20', 'sma_50', 'volatility',
    'momentum', 'volume_ratio', 'trend_1h', 'volume_spike_4h'
]

# === 4. Загрузка моделей ===
stacked = joblib.load(MODEL_PATH)
lstm_model = load_model(LSTM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === 5. Подготовка данных ===
if len(df_15m) < SEQUENCE_LENGTH:
    raise ValueError("Недостаточно данных")

X_seq, X_flat = [], []
for i in range(SEQUENCE_LENGTH, len(df_15m)):
    X_seq.append(df_15m[features].iloc[i - SEQUENCE_LENGTH:i].values)
    X_flat.append(df_15m[features].iloc[i].values)

X_seq = np.array(X_seq)
X_flat = np.array(X_flat)

# === 6. Бэкте́ст ===
trades = []
capital = INITIAL_CAPITAL
last_exit_idx = -1

X_seq_scaled = scaler.transform(
    X_seq.reshape(-1, X_seq.shape[-1])
).reshape(X_seq.shape)

for i in range(len(X_seq)):
    current_idx = SEQUENCE_LENGTH + i
    if current_idx < last_exit_idx:
        continue

    try:
        lstm_proba = lstm_model.predict(X_seq_scaled[i:i+1], verbose=0).flatten()[0]
        X_sample = pd.DataFrame([X_flat[i]], columns=stacked['features'])
        xgb_proba = stacked['xgb_model'].predict_proba(X_sample)[0, 1]
        stacked_input = np.array([[lstm_proba, xgb_proba]])
        stacked_proba = stacked['meta_model'].predict_proba(stacked_input)[0, 1]

        if (
            stacked_proba > 0.8 and
            df_15m.iloc[current_idx]['trend_1h'] == 1 and
            df_15m.iloc[current_idx]['volume_spike_4h'] == 1
        ):
            entry_idx = current_idx
            exit_idx = entry_idx + HOLD_CANDLES
            if exit_idx >= len(df_15m): continue

            entry_price = df_15m.iloc[entry_idx]["close"]
            exit_price = df_15m.iloc[exit_idx]["close"]
            # SHORT: прибыль = (entry - exit) / entry
            pnl_gross = (entry_price - exit_price) / entry_price
            pnl_net = pnl_gross - COMMISSION
            capital *= (1 + pnl_net)

            trades.append({
                "entry_time": df_15m.iloc[entry_idx]["timestamp"],
                "exit_time": df_15m.iloc[exit_idx]["timestamp"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_net": pnl_net
            })
            last_exit_idx = exit_idx
    except Exception as e:
        print(f"⚠️ Ошибка: {e}")

# === 7. Анализ ===
if trades:
    df_trades = pd.DataFrame(trades)
    win_rate = (df_trades["pnl_net"] > 0).mean()
    total_return = (capital / INITIAL_CAPITAL - 1) * 100
    avg_pnl = df_trades["pnl_net"].mean()
    max_dd = df_trades["pnl_net"].min()

    print(f"\n📊 Результаты: Win Rate={win_rate:.1%}, Доходность={total_return:+.2f}%, Капитал=${capital:.2f}")
    df_trades.to_csv(LOG_FILE, index=False)
    print(f"✅ Лог сохранён: {LOG_FILE}")
else:
    print("❌ Нет сделок")

print("✅ Бэкте́ст SHORT завершён")