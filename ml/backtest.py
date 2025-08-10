# ml/backtest.py
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
import joblib
import os

# Пути
MODEL_PATH = "./models/btc_long_model.pkl"
DATA_3M_PATH = "./data/processed/btc_usdt_3m.parquet"
LOG_FILE = "./logs/simple_backtest_log.csv"

# Параметры
WINDOW_TRAIN_HOURS = 30
VALIDATION_HOURS = 12
STEP_HOURS = 3
HOLD_CANDLES = 12
COMMISSION = 0.0008  # 0.08%
INITIAL_CAPITAL = 1000.0

print("🔍 Запуск бэктеста стратегии LONG...")

# === 1. Загружаем данные ===
if not os.path.exists(DATA_3M_PATH):
    raise FileNotFoundError(f"Файл не найден: {DATA_3M_PATH}")

df = pd.read_parquet(DATA_3M_PATH)
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"✅ Загружено {len(df)} свечей")

# Преобразуем в минуты
df['timestamp_min'] = (df['timestamp_ms'] // 60_000).astype(int)

# === 2. Загружаем модель ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print(f"✅ Модель загружена: {MODEL_PATH}")

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
features = [
    'close', 'volume', 'sma_20', 'sma_50', 'volatility',
    'momentum', 'volume_ratio'
]

# === 4. Бэктест ===
trades = []
log_entries = []

window_minutes = int(WINDOW_TRAIN_HOURS * 60)
step_minutes = int(STEP_HOURS * 60)
val_minutes = int(VALIDATION_HOURS * 60)

current_idx = window_minutes
capital = INITIAL_CAPITAL
equity_curve = [capital]

while current_idx < len(df) - HOLD_CANDLES:
    try:
        end_time = df.iloc[current_idx]['timestamp_min']
        start_time = end_time - window_minutes

        # Обучающая выборка
        train_mask = (df['timestamp_min'] >= start_time) & (df['timestamp_min'] < end_time)
        X_train = df[train_mask][features]
        if len(X_train) == 0 or X_train.isna().any().any():
            current_idx += step_minutes
            continue

        # Валидационная выборка
        val_mask = (df['timestamp_min'] >= end_time) & (df['timestamp_min'] < end_time + val_minutes)
        X_val = df[val_mask][features]
        if len(X_val) == 0 or X_val.isna().any().any():
            current_idx += step_minutes
            continue

        # Прогноз
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba > 0.6).astype(int)

        # Регистрируем все сделки
        for i, signal in enumerate(y_pred):
            if signal == 1:
                entry_idx = X_val.index[i]
                entry_price = df.loc[entry_idx, "close"]
                exit_idx = min(entry_idx + HOLD_CANDLES, len(df) - 1)
                exit_price = df.loc[exit_idx, "close"]
                pnl_gross = (exit_price - entry_price) / entry_price
                pnl_net = pnl_gross - COMMISSION
                capital *= (1 + pnl_net)

                trades.append({
                    "entry_time": df.loc[entry_idx, "timestamp"],
                    "exit_time": df.loc[exit_idx, "timestamp"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_gross": pnl_gross,
                    "pnl_net": pnl_net
                })

        equity_curve.append(capital)
        current_idx += step_minutes

    except Exception as e:
        print(f"❌ Ошибка на шаге {current_idx}: {e}")
        current_idx += step_minutes
        continue

# === 5. Анализ результатов ===
if trades:
    df_trades = pd.DataFrame(trades)
    win_rate = (df_trades["pnl_net"] > 0).mean()
    total_return = (capital / INITIAL_CAPITAL - 1) * 100
    avg_pnl = df_trades["pnl_net"].mean()
    max_drawdown = df_trades["pnl_net"].min()

    print(f"\n📊 Результаты бэктеста:")
    print(f"  Всего сделок: {len(df_trades)}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Средняя прибыль (чистая): {avg_pnl:+.2%}")
    print(f"  Общая доходность: {total_return:+.2f}%")
    print(f"  Конечный капитал: ${capital:.2f}")
    print(f"  Макс. просадка: {max_drawdown:+.2%}")
else:
    print("❌ Нет сделок для анализа")

# === 6. Сохранение лога ===
os.makedirs("./logs", exist_ok=True)
if trades:
    log_df = pd.DataFrame(trades)
    log_df.to_csv(LOG_FILE, index=False)
    print(f"✅ Лог сохранён: {LOG_FILE}")

print("✅ Бэктест завершён")