# fetcher/smart_fetcher.py
from datetime import datetime, timedelta
import pandas as pd
import httpx
import time
import os

# --- Настройки ---
url = "https://www.okx.com/api/v5/market/history-candles"
instId = "BTC-USDT"
bar = "3m"
limit = "300"  # Должно быть строкой
days_history = 45

# --- Определяем диапазон ---
end_dt = datetime.now() - timedelta(minutes=10)  # Защита от будущего
start_dt = end_dt - timedelta(days=days_history)

# Преобразуем в миллисекунды
end_ms = int(end_dt.timestamp() * 1000)
start_ms = int(start_dt.timestamp() * 1000)

# --- Сбор данных ---
all_candles = []
request_count = 0
max_requests = 300  # Ограничение на количество запросов

print(f"🚀 Запуск смарт-фетчера")
print(f"📅 Диапазон: {start_dt.strftime('%Y-%m-%d %H:%M')} → {end_dt.strftime('%Y-%m-%d %H:%M')}")
print(f"📊 Таймфрейм: {bar}, лимит: {limit} свечей/запрос")

while end_ms > start_ms and request_count < max_requests:
    # Формируем параметры запроса
    params = {
        "instId": instId,
        "bar": bar,
        "before": str(end_ms),
        "limit": limit
    }

    try:
        # Выполняем запрос
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json().get("data", [])

        # Если данных нет — выходим
        if not data:
            print("ℹ️ Достигнут конец доступных данных")
            break

        # Обрабатываем свечи
        batch = []
        for candle in data:
            ts = int(candle[0])
            if ts < start_ms:  # Игнорируем свечи вне диапазона
                continue
            batch.append({
                "timestamp_ms": ts,
                "timestamp": datetime.fromtimestamp(ts / 1000),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume_btc": float(candle[5]),
                "volume_usdt": float(candle[6]),
            })

        # Если нет подходящих свечей — выходим
        if not batch:
            print("ℹ️ Нет новых свечей в диапазоне")
            break

        # Добавляем в начало (т.к. API возвращает от новых к старым)
        all_candles = batch + all_candles
        print(f"✅ Получено {len(batch)} свечей (запрос #{request_count + 1})")

        # Обновляем курсор: на 1 ms меньше самой старой свечи
        oldest_timestamp = int(data[-1][0])
        end_ms = oldest_timestamp - 1

        request_count += 1
        time.sleep(0.15)  # Ограничение скорости (OKX: 20 запросов/2 сек)

    except Exception as e:
        print(f"❌ Ошибка при запросе #{request_count + 1}: {str(e)}")
        break

# --- Сохранение результата ---
if all_candles:
    df = pd.DataFrame(all_candles)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Создаём папку, если не существует
    os.makedirs("data/processed", exist_ok=True)
    filepath = "data/processed/btc_usdt_3m.parquet"
    df.to_parquet(filepath, index=False)

    # Статистика
    first_time = df["timestamp"].min().strftime("%Y-%m-%d %H:%M")
    last_time = df["timestamp"].max().strftime("%Y-%m-%d %H:%M")
    total = len(df)

    print(f"\n✅ Успешно загружено {total} свечей")
    print(f"📅 Диапазон: {first_time} — {last_time}")
    print(f"📁 Сохранено в: {filepath}")
else:
    print("❌ Нет данных для сохранения")