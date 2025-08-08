import pandas as pd
import requests
import os
import time
from datetime import datetime, timedelta
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Конфигурация
SYMBOL = "BTC-USDT"
TIMEFRAME = "3m"
DEEP_DAYS = 2
LIMIT_PER_REQUEST = 100
REQUEST_DELAY = 0.5  # Увеличили задержку для надежности
MAX_REQUESTS = 50

def fetch_candles(symbol, timeframe, start_ts, end_ts, max_requests=MAX_REQUESTS):
    """Загружает свечи используя более надежный метод пагинации"""
    candles = []
    request_count = 0
    last_ts = end_ts
    
    while request_count < max_requests:
        request_count += 1
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={symbol}&bar={timeframe}&before={last_ts}&limit={LIMIT_PER_REQUEST}"
        
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                logger.warning(f"HTTP error {response.status_code}")
                time.sleep(2)
                continue
                
            data = response.json().get("data", [])
            
            if not data:
                logger.info("No more data available")
                break
                
            # Обрабатываем свечи
            new_candles = []
            for candle in data:
                try:
                    ts = int(candle[0])
                    if ts < start_ts:
                        continue
                        
                    new_candles.append({
                        "timestamp_ms": ts,
                        "timestamp": datetime.utcfromtimestamp(ts / 1000),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume_btc": float(candle[5]),
                        "volume_usdt": float(candle[6]),
                    })
                except (IndexError, ValueError) as e:
                    logger.warning(f"Invalid candle format: {candle} - {str(e)}")
            
            if not new_candles:
                logger.info("No new candles in response")
                break
                
            # Добавляем новые свечи
            candles.extend(new_candles)
            
            # Получаем timestamp самой старой свечи в ответе
            oldest_ts = min([c["timestamp_ms"] for c in new_candles])
            
            # Проверяем прогресс
            if oldest_ts >= last_ts:
                logger.warning("Pagination error: timestamp not decreasing")
                # Пытаемся сдвинуться на размер лимита
                last_ts = oldest_ts - (180000 * LIMIT_PER_REQUEST)
            else:
                last_ts = oldest_ts
                
            logger.info(f"Request #{request_count}: got {len(new_candles)} candles, oldest: {datetime.utcfromtimestamp(oldest_ts/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            time.sleep(3)
            
        time.sleep(REQUEST_DELAY)
    
    return candles

def main():
    # Рассчитываем временные границы
    end_time = datetime.utcnow() - timedelta(minutes=3)  # Последняя завершенная свеча
    start_time = end_time - timedelta(days=DEEP_DAYS)
    
    logger.info(f"🔍 Loading {DEEP_DAYS} days of data")
    logger.info(f"🔍 Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🔍 End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Конвертируем в timestamp в миллисекундах
    end_ts = int(end_time.timestamp() * 1000)
    start_ts = int(start_time.timestamp() * 1000)
    
    # Загружаем данные
    candles = fetch_candles(SYMBOL, TIMEFRAME, start_ts, end_ts)
    
    if not candles:
        logger.error("❌ No candles loaded")
        return
        
    # Создаем DataFrame
    df = pd.DataFrame(candles)
    
    # Удаляем дубликаты
    initial_count = len(df)
    df = df.drop_duplicates("timestamp_ms")
    if initial_count != len(df):
        logger.warning(f"Removed {initial_count - len(df)} duplicates")
    
    # Сортируем по времени
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Фильтруем по диапазону
    df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]
    
    # Сохраняем данные
    os.makedirs("data/processed", exist_ok=True)
    filepath = f"data/processed/{SYMBOL.lower().replace('-', '_')}_{TIMEFRAME}.parquet"
    df.to_parquet(filepath, index=False)
    
    # Отчет
    logger.info(f"✅ Successfully saved {len(df)} candles")
    logger.info(f"📅 First candle: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📅 Last candle: {df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"💾 File: {filepath}")
    
    # Проверка полноты данных
    expected_candles = DEEP_DAYS * 24 * 60 // 3
    if len(df) < expected_candles:
        logger.warning(f"⚠️ Warning: loaded {len(df)} of {expected_candles} expected candles")
    else:
        logger.info(f"📊 Data completeness: 100% ({len(df)}/{expected_candles} candles)")
    
    # Логируем последние 5 свечей
    logger.info("\nLast 5 candles:")
    logger.info(df[["timestamp", "close"]].tail().to_string(index=False))

if __name__ == "__main__":
    main()