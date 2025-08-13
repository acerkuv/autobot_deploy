# ml/retrain_backtest_short.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os
from datetime import timedelta

# === Пути ===
DATA_15M_PATH = "./data/processed/btc_usdt_15m.parquet"
DATA_1H_PATH = "./data/processed/btc_usdt_1h.parquet"
DATA_4H_PATH = "./data/processed/btc_usdt_4h.parquet"
LOG_FILE = "./logs/retrain_backtest_short_log.csv"

# Пути к предобученным экстракторам
LSTM_EXTRACTOR_PATH = "./models/lstm_extractor_v2_short.keras"
XGB_EXTRACTOR_PATH = "./models/xgb_extractor_v2_short.pkl"
SCALER_PATH = "./models/scaler_15m_short.pkl"

# Авто-сохранение валидных моделей
VALID_MODEL_DIR = "./models/valid_weekly_short/"
os.makedirs(VALID_MODEL_DIR, exist_ok=True)

# === Параметры (должны совпадать с обучением!) ===
SEQUENCE_LENGTH = 90  # как в stacked_model_v2_short_retrain.py
HOLD_CANDLES = 4
COMMISSION = 0.0008
INITIAL_CAPITAL = 1000.0
RETRAIN_INTERVAL_DAYS = 7
MIN_TRAIN_DAYS = 90
AUC_THRESHOLD = 0.65

print("🔍 Запуск бэкте́ста с дообучением (SHORT, раз в неделю)")

# === 1. Проверка наличия файлов ===
for path in [DATA_15M_PATH, DATA_1H_PATH, DATA_4H_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

# === 2. Загрузка данных ===
df_15m = pd.read_parquet(DATA_15M_PATH)
df_1h = pd.read_parquet(DATA_1H_PATH)
df_4h = pd.read_parquet(DATA_4H_PATH)

df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])

df_15m = df_15m.sort_values("timestamp").reset_index(drop=True)
df_1h = df_1h.sort_values("timestamp").reset_index(drop=True)
df_4h = df_4h.sort_values("timestamp").reset_index(drop=True)

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

df_15m = add_features(df_15m)
df_1h = add_features(df_1h)
df_4h = add_features(df_4h)

# === 4. Мультитаймфреймовые фичи (для SHORT) ===
df_15m['trend_1h'] = df_15m['timestamp'].map(
    df_1h.set_index('timestamp')['close'] < df_1h.set_index('timestamp')['sma_50']  # медвежий тренд
)
df_15m['volume_spike_4h'] = df_15m['timestamp'].map(
    df_4h.set_index('timestamp')['volume'] > df_4h.set_index('timestamp')['volume'].rolling(10).mean() * 1.5
)

df_15m['trend_1h'] = pd.to_numeric(df_15m['trend_1h'], errors='coerce').ffill().bfill().astype(int)
df_15m['volume_spike_4h'] = pd.to_numeric(df_15m['volume_spike_4h'], errors='coerce').ffill().bfill().astype(int)

# === 5. Фичи (должны совпадать с обучением!) ===
features = [
    'close', 'volume', 'sma_20', 'sma_50', 'volatility',
    'momentum', 'volume_ma', 'volume_ratio', 'trend_1h', 'volume_spike_4h'
]

# === 6. Загрузка предобученных экстракторов ===
print("🔁 Загрузка предобученных экстракторов...")

# Проверка файлов
for path in [LSTM_EXTRACTOR_PATH, XGB_EXTRACTOR_PATH, SCALER_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

lstm_extractor = load_model(LSTM_EXTRACTOR_PATH)
xgb_extractor = joblib.load(XGB_EXTRACTOR_PATH)
scaler = joblib.load(SCALER_PATH)

# Эмбеддер (предпоследний слой LSTM)
intermediate_layer_model = Model(
    inputs=lstm_extractor.input,
    outputs=lstm_extractor.layers[-2].output
)

# === 7. Подготовка данных для бэкте́ста ===
trades = []
capital = INITIAL_CAPITAL
last_exit_idx = -1

start_date = df_15m['timestamp'].iloc[SEQUENCE_LENGTH]
end_date = df_15m['timestamp'].iloc[-1]

current_date = start_date
week_num = 1
last_valid_meta_model = None
last_valid_week = -1

# === 8. Основной цикл: неделя за неделей ===
while current_date < end_date:
    print(f"\n🔄 Неделя {week_num}: {current_date.date()}")

    # Окна: обучение (90 дней до текущей даты), тест (следующая неделя)
    train_start = current_date - timedelta(days=MIN_TRAIN_DAYS)
    train_end = current_date
    test_end = current_date + timedelta(days=RETRAIN_INTERVAL_DAYS)

    train_mask = (df_15m['timestamp'] >= train_start) & (df_15m['timestamp'] < train_end)
    test_mask = (df_15m['timestamp'] >= train_end) & (df_15m['timestamp'] < test_end)

    df_train = df_15m[train_mask].copy()
    df_test = df_15m[test_mask].copy()

    if len(df_train) < SEQUENCE_LENGTH or len(df_test) == 0:
        current_date += timedelta(days=RETRAIN_INTERVAL_DAYS)
        week_num += 1
        continue

    # Целевая переменная: цена упадёт ниже на 0.3%
    TARGET_THRESHOLD = 0.997
    df_train['future_close'] = df_train['close'].shift(-HOLD_CANDLES)
    df_train['target'] = (
        (df_train['future_close'] < df_train['close'] * TARGET_THRESHOLD) &
        (df_train['trend_1h'] == 1)
    ).astype(int)
    df_train.dropna(subset=features + ['target'], inplace=True)
    df_train = df_train.reset_index(drop=True)

    # Подготовка последовательностей
    def create_sequences(data, seq_len, features, target_col):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[features].iloc[i-seq_len:i].values)
            y.append(data[target_col].iloc[i])
        return np.array(X), np.array(y)

    X_seq, y_seq = create_sequences(df_train, SEQUENCE_LENGTH, features, 'target')
    X_flat = df_train[features].values[SEQUENCE_LENGTH:]
    y_flat = df_train['target'].values[SEQUENCE_LENGTH:]

    if len(X_seq) == 0:
        current_date += timedelta(days=RETRAIN_INTERVAL_DAYS)
        week_num += 1
        continue

    # Нормализация
    X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_seq_scaled = scaler.transform(X_seq_flat).reshape(X_seq.shape)

    # Извлечение признаков
    lstm_embeddings = intermediate_layer_model.predict(X_seq_scaled, verbose=0)
    xgb_proba = xgb_extractor.predict_proba(X_flat)[:, 1]
    X_stack = np.column_stack([lstm_embeddings, xgb_proba])
    y_stack = y_seq

    split_idx = int(0.8 * len(X_stack))
    X_train, X_val = X_stack[:split_idx], X_stack[split_idx:]
    y_train, y_val = y_stack[:split_idx], y_stack[split_idx:]

    # Дообучение мета-модели
    meta_model = LGBMClassifier(n_estimators=50, random_state=42)
    meta_model.fit(X_train, y_train)

    # Валидация
    try:
        y_proba = meta_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
    except:
        auc = 0.0

    is_valid = auc >= AUC_THRESHOLD
    print(f"  📊 AUC = {auc:.3f} → {'✅ Валидна' if is_valid else '❌ Не валидна'}")

    # Выбор модели
    trading_model = None
    if is_valid:
        trading_model = meta_model
        last_valid_meta_model = meta_model
        last_valid_week = week_num
        joblib.dump(meta_model, f"{VALID_MODEL_DIR}/meta_model_week_{week_num}.pkl")
        print(f"  💾 Модель сохранена как валидная")
    elif last_valid_meta_model is not None:
        trading_model = last_valid_meta_model
        print(f"  🛑 Используем последнюю валидную модель (неделя {last_valid_week})")
    else:
        print(f"  🛑 Нет валидной модели → холд")
        current_date += timedelta(days=RETRAIN_INTERVAL_DAYS)
        week_num += 1
        continue

    # === Бэкте́ст ===
    for i in range(len(df_test)):
        try:
            current_idx = df_15m.index[df_15m['timestamp'] == df_test.iloc[i]['timestamp']].tolist()[0]
            if current_idx < last_exit_idx:
                continue

            if current_idx < SEQUENCE_LENGTH:
                continue

            # Подготовка последовательности: (90, 10)
            seq_data = df_15m[features].iloc[current_idx - SEQUENCE_LENGTH:current_idx].values

            # ✅ Правильная нормализация
            seq_scaled = scaler.transform(seq_data).reshape(1, SEQUENCE_LENGTH, len(features))

            # Прогнозы
            lstm_emb = intermediate_layer_model.predict(seq_scaled, verbose=0).flatten()
            flat_data = df_15m[features].iloc[current_idx].values
            xgb_prob = xgb_extractor.predict_proba([flat_data])[0, 1]
            stacked_input = np.array([[*lstm_emb, xgb_prob]])

            stacked_proba = trading_model.predict_proba(stacked_input)[0, 1]

            # Сигнал SHORT
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
                pnl_net = (entry_price - exit_price) / exit_price - COMMISSION  # SHORT: profit when price drops
                capital *= (1 + pnl_net)

                trades.append({
                    "entry_time": df_15m.iloc[entry_idx]["timestamp"],
                    "exit_time": df_15m.iloc[exit_idx]["timestamp"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_net": pnl_net,
                    "week": week_num,
                    "model_source": "current" if is_valid else "fallback"
                })
                last_exit_idx = exit_idx

        except Exception as e:
            print(f"  ⚠️ Ошибка: {e}")
            continue

    current_date += timedelta(days=RETRAIN_INTERVAL_DAYS)
    week_num += 1

# === 9. Анализ результатов ===
if trades:
    df_trades = pd.DataFrame(trades)
    win_rate = (df_trades["pnl_net"] > 0).mean()
    total_return = (capital / INITIAL_CAPITAL - 1) * 100
    avg_pnl = df_trades["pnl_net"].mean()
    max_dd = df_trades["pnl_net"].min()

    print(f"\n📊 Финальные результаты (SHORT):")
    print(f"  Всего сделок: {len(df_trades)}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Средняя прибыль: {avg_pnl:+.2%}")
    print(f"  Общая доходность: {total_return:+.2f}%")
    print(f"  Конечный капитал: ${capital:.2f}")
    print(f"  Макс. просадка: {max_dd:+.2%}")

    df_trades.to_csv(LOG_FILE, index=False)
    print(f"✅ Лог сохранён: {LOG_FILE}")
else:
    print("❌ Нет сделок — стратегия в холде")

print("✅ Бэкте́ст с дообучением (SHORT) завершён")