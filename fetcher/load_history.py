# fetcher/load_history.py
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

def fetch_okx_candles(inst_id, bar, days_back=45):
    """
    Загружает исторические свечи с OKX API.
    Использует пагинацию по `before` для стабильной загрузки.
    """
    # Определяем временные границы
    end_time = datetime.utcnow() - timedelta(minutes=10)  # Защита от будущего
    start_time = end_time - timedelta(days=days_back)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    # Базовый URL API (без пробелов!)
    url = "https://www.okx.com/api/v5/market/history-candles"

    # Настройки таймфрейма
    timeframe_ms = {"3m": 180000, "1m": 60000, "1h": 3600000}.get(bar)
    if not timeframe_ms:
        raise ValueError(f"Unsupported timeframe: {bar}. Use '1m', '3m', or '1h'.")

    print(f"🚀 Загрузка {days_back} дней истории {inst_id} ({bar})")
    print(f"📅 Диапазон: {start_time.strftime('%Y-%m-%d %H:%M')} → {end_time.strftime('%Y-%m-%d %H:%M')}")

    all_candles = []
    current_before = end_ms
    request_count = 0
    max_requests = 200  # Ограничение на случай ошибки

    while current_before > start_ms and request_count < max_requests:
        params = {
            "instId": inst_id,
            "bar": bar,
            "before": str(current_before),
            "limit": "100"
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                break

            data = response.json()
            if data.get("code") != "0":
                print(f"❌ API error: {data.get('msg', 'Unknown error')}")
                break

            candles_data = data.get("data", [])
            if not candles_data:
                print("ℹ️ Нет новых данных")
                break

            # Обработка свечей
            batch = []
            for candle in candles_data:
                ts = int(candle[0])
                if ts < start_ms:
                    continue  # Игнорируем вне диапазона
                batch.append({
                    "timestamp_ms": ts,
                    "timestamp": datetime.utcfromtimestamp(ts / 1000),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume_btc": float(candle[5]),
                    "volume_usdt": float(candle[6]),
                })

            if not batch:
                print("ℹ️ Нет свечей в диапазоне")
                break

            # Добавляем в начало (API возвращает от новых к старым)
            all_candles = batch + all_candles
            print(f"✅ Получено {len(batch)} свечей (всего: {len(all_candles)})")

            # Обновляем курсор: на 1 ms меньше последней (самой старой) свечи
            last_candle_timestamp = int(candles_data[-1][0])
            current_before = last_candle_timestamp - 1

            request_count += 1
            time.sleep(0.2)  # Лимит OKX: ~5 запросов/сек

        except Exception as e:
            print(f"❌ Ошибка при запросе #{request_count + 1}: {str(e)}")
            break

    if not all_candles:
        print("❌ Нет данных для сохранения")
        return pd.DataFrame()

    # Создаём DataFrame
    df = pd.DataFrame(all_candles)

    # Удаляем дубликаты по timestamp_ms
    df = df.drop_duplicates(subset=["timestamp_ms"], keep="last")

    # Сортируем по времени (от старых к новым)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    # Добавляем метаданные
    df["instrument"] = inst_id
    df["timeframe"] = bar

    return df

def save_to_parquet(df, filepath):
    """Сохраняет DataFrame в Parquet с созданием директории."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, index=False)
    print(f"✅ Сохранено {len(df)} свечей в {filepath}")

if __name__ == "__main__":
    # Конфигурация
    INSTRUMENT = "BTC-USDT"
    TIMEFRAME = "3m"
    DAYS_BACK = 45
    OUTPUT_FILE = f"./data/processed/{INSTRUMENT}_{TIMEFRAME}_last_{DAYS_BACK}_days.parquet"

    # Выполняем загрузку
    df = fetch_okx_candles(INSTRUMENT, TIMEFRAME, DAYS_BACK)

    if not df.empty:
        save_to_parquet(df, OUTPUT_FILE)
    else:
        print("❌ Загрузка не удалась")