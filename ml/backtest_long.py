# ml/backtest_long.py
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import joblib
import os

# === Пути ===
PROJECT_ROOT = "."
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "btc_usdt_3m.parquet")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "backtest_log.csv")

# === Параметры ===
WINDOW_TRAIN_HOURS = 30    # Обучение на 30 часах
VALIDATION_HOURS = 12      # Валидация на следующих 12 часах
STEP_HOURS = 3             # Шаг: +3 часа
HOLD_CANDLES = 12          # Держим сделку 12 свечей (36 минут)
MIN_DATA_FOR_TRAIN = 600   # Минимум 600 свечей для обучения
CONFIDENCE_THRESHOLD = 0.85 # Высокий порог уверенности
COMMISSION = 0.0008         # Комиссия Binance: 0.04% + 0.04%
INITIAL_CAPITAL = 1000.0    # Начальный капитал

print("🔍 Запуск ДЛИННОГО бэкте́ста с переобучением и валидацией...")

# === 1. Загружаем данные ===
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Файл не найден: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"✅ Загружено {len(df)} свечей (3m)")

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

df = add_features(df)

# === Загружаем 1h данные для фильтра по тренду ===
DATA_1H_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "btc_usdt_1h.parquet")
if os.path.exists(DATA_1H_PATH):
    df_1h = pd.read_parquet(DATA_1H_PATH)
    df_1h = df_1h.sort_values("timestamp").reset_index(drop=True)
    print(f"✅ Загружено {len(df_1h)} свечей (1h)")

    # Привязываем тренд 1h к 3m
    df_1h['hour'] = pd.to_datetime(df_1h['timestamp'], unit='ms').dt.floor('H')
    df['hour'] = pd.to_datetime(df['timestamp'], unit='ms').dt.floor('H')
    df = df.merge(df_1h[['hour', 'close']], on='hour', suffixes=('', '_1h'), how='left')
    df['trend_1h'] = (df['close'] > df['close_1h'].rolling(20).mean()).astype(int)
    df['trend_1h'] = df['trend_1h'].fillna(1).astype(int)  # Если нет данных — считаем тренд восходящим
else:
    print("⚠️ Файл 1h данных не найден. Продолжаем без фильтра по тренду.")
    df['trend_1h'] = 1  # По умолчанию разрешаем все сделки

# Фичи — должны быть ТОЧНО такими же, как в train.py
features = [
    'close', 'volume', 'sma_20', 'sma_50', 'volatility',
    'momentum', 'volume_ratio'
]

# === 3. Целевая переменная: +0.2% за 12 свечей (36 мин) ===
df['future_close'] = df['close'].shift(-HOLD_CANDLES)
df['target'] = (df['future_close'] > df['close'] * 1.002).astype(int)

# Удаляем NaN
df.dropna(subset=features + ['target', 'trend_1h'], inplace=True)
df = df.reset_index(drop=True)

print(f"📊 После очистки: {len(df)} свечей")

# === 4. Подготовка для бэкте́ста ===
trades = []
signals = []
log_entries = []

# Количество свечей за час: 20 (т.к. 3m → 20 свечей в час)
CANDLES_PER_HOUR = 20
step_candles = STEP_HOURS * CANDLES_PER_HOUR
window_candles = WINDOW_TRAIN_HOURS * CANDLES_PER_HOUR
val_candles = VALIDATION_HOURS * CANDLES_PER_HOUR

# Начинаем с момента, когда можно обучиться
current_idx = window_candles

# Счётчики
total_signals = 0
valid_windows = 0
equity_curve = [INITIAL_CAPITAL]  # Кривая капитала

while current_idx < len(df) - val_candles:
    try:
        # Обучающий диапазон: 30 часов до current_idx
        train_start_idx = current_idx - window_candles
        X_train = df.iloc[train_start_idx:current_idx][features]
        y_train = df.iloc[train_start_idx:current_idx]['target']

        if len(X_train) < MIN_DATA_FOR_TRAIN:
            print(f"⚠️ Недостаточно данных для обучения: {len(X_train)}")
            current_idx += step_candles
            continue

        # Валидационный диапазон: следующие 12 часов
        val_end_idx = min(current_idx + val_candles, len(df))
        X_val = df.iloc[current_idx:val_end_idx][features]
        y_val = df.iloc[current_idx:val_end_idx]['target']

        if len(X_val) == 0:
            current_idx += step_candles
            continue

        # Проверка, что есть положительные метки
        if y_train.sum() == 0:
            print("⚠️ Нет положительных меток в обучении")
            current_idx += step_candles
            continue

        # Вес для балансировки
        pos_weight = len(y_train) / y_train.sum()

        # === Одна модель: LightGBM (стабильная и быстрая) ===
        model = LGBMClassifier(
            n_estimators=100,
            random_state=42,
            scale_pos_weight=pos_weight,
            min_child_samples=20,
            objective='binary',
            metric='binary_logloss'
        )
        model.fit(X_train, y_train)

        # Прогноз
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba > CONFIDENCE_THRESHOLD).astype(int)

        # 🔍 Диагностика
        print(f"🔹 Валидация: {df.iloc[current_idx]['timestamp']} → {df.iloc[val_end_idx-1]['timestamp']}")
        print(f"🔹 X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
        print(f"🔹 y_train sum: {y_train.sum()}, y_val sum: {y_val.sum()}")

        # 📊 Валидация: метрики
        if y_val.sum() > 0:
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            auc = roc_auc_score(y_val, y_proba)
            print(f"📊 Валидация: Prec={prec:.3f}, Rec={rec:.3f}, AUC={auc:.3f}")
        else:
            print("⚠️ Нет положительных меток в валидации для расчёта метрик")

        print(f"🔹 Сигналов (confidence > {CONFIDENCE_THRESHOLD}): {y_pred.sum()} из {len(y_pred)}")
        total_signals += y_pred.sum()

        # Регистрируем сделки
        capital = equity_curve[-1]
        for i, signal in enumerate(y_pred):
            if signal == 1:
                entry_idx = current_idx + i
                # ✅ Фильтр: только если тренд 1h восходящий
                if df.iloc[entry_idx]['trend_1h'] == 0:
                    continue  # Пропускаем, если тренд не восходящий

                entry_price = df.iloc[entry_idx]["close"]
                exit_idx = min(entry_idx + HOLD_CANDLES, len(df) - 1)
                exit_price = df.iloc[exit_idx]["close"]
                pnl_gross = (exit_price - entry_price) / entry_price
                pnl_net = pnl_gross - COMMISSION  # Учёт комиссии
                capital *= (1 + pnl_net)

                trades.append({
                    "entry_time": df.iloc[entry_idx]["timestamp"],
                    "exit_time": df.iloc[exit_idx]["timestamp"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_gross": pnl_gross,
                    "pnl_net": pnl_net
                })

        equity_curve.append(capital)

        # Лог для анализа
        log_entries.append({
            "train_start": df.iloc[train_start_idx]["timestamp"],
            "train_end": df.iloc[current_idx - 1]["timestamp"],
            "val_start": df.iloc[current_idx]["timestamp"],
            "val_end": df.iloc[val_end_idx - 1]["timestamp"],
            "train_size": len(X_train),
            "val_size": len(X_val),
            "pos_weight": pos_weight,
            "signals": y_pred.sum(),
            "capital": capital
        })

        valid_windows += 1
        current_idx += step_candles

    except Exception as e:
        print(f"❌ Ошибка на шаге {current_idx}: {e}")
        current_idx += step_candles
        continue

# === 5. Анализ результатов ===
print(f"\n📊 Всего обработано окон: {valid_windows}, всего сигналов: {total_signals}")

if trades:
    df_trades = pd.DataFrame(trades)
    win_rate = (df_trades["pnl_net"] > 0).mean()
    total_return = (equity_curve[-1] / INITIAL_CAPITAL - 1) * 100
    avg_pnl = df_trades["pnl_net"].mean()
    max_drawdown = df_trades["pnl_net"].min()

    print(f"\n📊 Результаты ДЛИННОГО бэкте́ста:")
    print(f"  Всего сделок: {len(df_trades)}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Средняя прибыль (чистая): {avg_pnl:+.2%}")
    print(f"  Общая доходность: {total_return:+.2f}%")
    print(f"  Конечный капитал: ${equity_curve[-1]:.2f}")
    print(f"  Макс. просадка (по сделкам): {max_drawdown:+.2%}")
else:
    print("❌ Нет сделок для анализа")

# === 6. Сохранение лога ===
os.makedirs(LOGS_DIR, exist_ok=True)
if log_entries:
    log_df = pd.DataFrame(log_entries)
    log_df.to_csv(LOG_FILE, index=False)
    print(f"✅ Лог сохранён: {LOG_FILE}")
else:
    print("⚠️ Нет данных для сохранения лога")

print("✅ Длинный бэкте́ст завершён")