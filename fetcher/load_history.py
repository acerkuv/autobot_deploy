# fetcher/load_history.py
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import os

def fetch_binance_candles(symbol, timeframe, days_back=200):
    """
    Загружает исторические свечи с Binance через ccxt.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    # ✅ Правильно вычисляем timestamp 200 дней назад
    since_date = datetime.utcnow() - timedelta(days=days_back)
    since = exchange.parse8601(since_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    end_time = exchange.milliseconds()
    all_candles = []

    print(f"🚀 Загрузка {days_back} дней {symbol} {timeframe} с Binance")

    while since < end_time:
        try:
            # Запрашиваем 1000 свечей за раз
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not candles:
                break

            all_candles.extend(candles)
            print(f"✅ Получено {len(candles)} свечей (всего: {len(all_candles)})")

            # Обновляем `since` на последнюю свечу
            since = candles[-1][0] + exchange.parse_timeframe(timeframe) * 1000

            # Соблюдаем лимиты
            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print(f"❌ Ошибка: {str(e)}")
            break

    if not all_candles:
        return pd.DataFrame()

    # Создаём DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df['instrument'] = symbol
    df['timeframe'] = timeframe

    # Удаляем дубликаты и сортируем
    df = df.drop_duplicates(subset=['timestamp_ms']).sort_values('timestamp_ms').reset_index(drop=True)

    return df

def save_to_parquet(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, index=False)
    print(f"✅ Сохранено {len(df)} свечей в {filepath}")

if __name__ == "__main__":
    SYMBOL = "BTC/USDT"
    TIMEFRAMES = ["15m", "1h", "4h"]
    DAYS_BACK = 200

    for tf in TIMEFRAMES:
        df = fetch_binance_candles(SYMBOL, tf, DAYS_BACK)
        if not df.empty:
            filepath = f"./data/processed/btc_usdt_{tf}.parquet"
            save_to_parquet(df, filepath)