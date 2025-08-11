# ml/backtest_stacked.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# === Пути ===
MODEL_PATH = "./models/btc_long_stacked_v2_model_15m.pkl"
LSTM_MODEL_PATH = "./models/lstm_model_v2_15m.keras"
DATA_15M_PATH = "./data/processed/btc_usdt_15m.parquet"
DATA_1H_PATH = "./data/processed/btc_usdt_1h.parquet"
DATA_4H_PATH = "./data/processed/btc_usdt_4h.parquet"
LOG_FILE = "./logs/backtest_stacked_log.csv"

# === Параметры (должны совпадать с обучением!) ===
SEQUENCE_LENGTH = 90      # Из stacked_model_v2.py
HOLD_CANDLES = 4          # 4 свечи × 15m = 60 минут
COMMISSION = 0.0008        # 0.08% (Binance)
INITIAL_CAPITAL = 1000.0   # $1000

print("🔍 Запуск бэкте́ста стекинг-модели (15m) с мультитаймфреймом...")

# === 1. Загружаем данные ===
for path, name in [
    (DATA_15M_PATH, "15m"),
    (DATA_1H_PATH, "1h"),
    (DATA_4H_PATH, "4h")
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

df_15m = pd.read_parquet(DATA_15M_PATH)
df_1h = pd.read_parquet(DATA_1H_PATH)
df_4h = pd.read_parquet(DATA_4H_PATH)

df_15m = df_15m.sort_values("timestamp").reset_index(drop=True)
df_1h = df_1h.sort_values("timestamp").reset_index(drop=True)
df_4h = df_4h.sort_values("timestamp").reset_index(drop=True)

print(f"✅ Загружено {len(df_15m)} свечей (15m)")
print(f"✅ Загружено {len(df_1h)} свечей (1h)")
print(f"✅ Загружено {len(df_4h)} свечей (4h)")

# === 2. Фичи для 15m ===
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

# === 3. Добавляем мультитаймфреймовые фичи ===
# Привязываем 1h и 4h к 15m по времени
df_15m['trend_1h'] = df_15m['timestamp'].map(
    df_1h.set_index('timestamp')['close'] > df_1h.set_index('timestamp')['sma_50']
)
df_15m['volume_spike_4h'] = df_15m['timestamp'].map(
    df_4h.set_index('timestamp')['volume'] > df_4h.set_index('timestamp')['volume'].rolling(10).mean() * 1.5
)

# ✅ Безопасное заполнение NaN
df_15m['trend_1h'] = pd.to_numeric(df_15m['trend_1h'], errors='coerce').ffill().bfill().astype(int)
df_15m['volume_spike_4h'] = pd.to_numeric(df_15m['volume_spike_4h'], errors='coerce').ffill().bfill().astype(int)

# === 4. Фичи для модели ===
features = [
    'close', 'volume', 'sma_20', 'sma_50', 'volatility',
    'momentum', 'volume_ratio', 'trend_1h', 'volume_spike_4h'
]

# === 5. Загружаем модели ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Мета-модель не найдена: {MODEL_PATH}")

if not os.path.exists(LSTM_MODEL_PATH):
    raise FileNotFoundError(f"Модель LSTM не найдена: {LSTM_MODEL_PATH}")

try:
    stacked = joblib.load(MODEL_PATH)
    print(f"✅ Мета-модель загружена: {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Ошибка загрузки мета-модели: {e}")

try:
    lstm_model = load_model(LSTM_MODEL_PATH)
    print(f"✅ Модель LSTM загружена: {LSTM_MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Ошибка загрузки LSTM: {e}")

# === 6. Подготовка данных для бэкте́ста ===
trades = []
capital = INITIAL_CAPITAL
equity_curve = [capital]

if len(df_15m) < SEQUENCE_LENGTH:
    raise ValueError(f"Недостаточно данных: {len(df_15m)} < {SEQUENCE_LENGTH}")

# Подготовка последовательностей
X_seq, X_flat = [], []
for i in range(SEQUENCE_LENGTH, len(df_15m)):
    X_seq.append(df_15m[features].iloc[i - SEQUENCE_LENGTH:i].values)
    X_flat.append(df_15m[features].iloc[i].values)

X_seq = np.array(X_seq)
X_flat = np.array(X_flat)

print(f"📊 Подготовлено {len(X_seq)} последовательностей для бэкте́ста")

# === 7. Бэкте́ст (с запретом перекрытия сделок) ===
last_exit_idx = -1  # Индекс, до которого нельзя открывать новые сделки

for i in range(len(X_seq)):
    try:
        # Текущий индекс в df_15m
        current_idx = SEQUENCE_LENGTH + i

        # Пропускаем, если уже есть активная сделка
        if current_idx < last_exit_idx:
            equity_curve.append(capital)
            continue

        # Прогноз LSTM
        lstm_proba = lstm_model.predict(X_seq[i:i+1], verbose=0).flatten()[0]

        # Прогноз XGBoost — с именами фичей (чтобы убрать предупреждение)
        X_sample = pd.DataFrame([X_flat[i]], columns=stacked['features'])
        xgb_proba = stacked['xgb_model'].predict_proba(X_sample)[0, 1]

        # Стекинг: LightGBM принимает только два признака — вероятности
        stacked_input = np.array([[lstm_proba, xgb_proba]])
        stacked_proba = stacked['meta_model'].predict_proba(stacked_input)[0, 1]

        # ✅ Сильный сигнал + фильтры по тренду и объёму
        if (
            stacked_proba > 0.8 and
            df_15m.iloc[current_idx]['trend_1h'] == 1 and
            df_15m.iloc[current_idx]['volume_spike_4h'] == 1
        ):
            entry_idx = current_idx
            exit_idx = entry_idx + HOLD_CANDLES

            # Проверка границ
            if exit_idx >= len(df_15m):
                equity_curve.append(capital)
                continue

            entry_price = df_15m.iloc[entry_idx]["close"]
            exit_price = df_15m.iloc[exit_idx]["close"]

            pnl_gross = (exit_price - entry_price) / entry_price
            pnl_net = pnl_gross - COMMISSION
            capital *= (1 + pnl_net)

            trades.append({
                "entry_time": df_15m.iloc[entry_idx]["timestamp"],
                "exit_time": df_15m.iloc[exit_idx]["timestamp"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_net": pnl_net
            })

            # ✅ Запрещаем новые сделки до выхода
            last_exit_idx = exit_idx

    except Exception as e:
        print(f"⚠️ Ошибка на шаге {i}: {e}")
        equity_curve.append(capital)
        continue

    equity_curve.append(capital)

# === 8. Анализ результатов ===
if trades:
    df_trades = pd.DataFrame(trades)
    win_rate = (df_trades["pnl_net"] > 0).mean()
    total_return = (capital / INITIAL_CAPITAL - 1) * 100
    avg_pnl = df_trades["pnl_net"].mean()
    max_drawdown = df_trades["pnl_net"].min()

    print(f"\n📊 Результаты бэкте́ста (15m + 1h + 4h):")
    print(f"  Всего сделок: {len(df_trades)}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Средняя прибыль (чистая): {avg_pnl:+.2%}")
    print(f"  Общая доходность: {total_return:+.2f}%")
    print(f"  Конечный капитал: ${capital:.2f}")
    print(f"  Макс. просадка: {max_drawdown:+.2%}")
else:
    print("❌ Нет сделок для анализа")

# === 9. Сохранение лога ===
os.makedirs("./logs", exist_ok=True)
if trades:
    df_trades.to_csv(LOG_FILE, index=False)
    print(f"✅ Лог сохранён: {LOG_FILE}")
else:
    print("⚠️ Нет сделок — лог не сохранён")

print("✅ Бэкте́ст завершён")