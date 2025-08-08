# shared/okx_api.py
import os
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from dateutil import tz
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import redis
import numpy as np

# === Загружаем .env из корня проекта ===
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ .env загружен из {env_path}")
else:
    print(f"❌ .env не найден: {env_path}")

# === Настройка часовых поясов ===
MSK_TZ = tz.gettz("Europe/Moscow")
UTC_TZ = timezone.utc

# === Логгер ===
logger = logging.getLogger("okx_fetcher")
logger.setLevel(logging.INFO)

def setup_logging():
    """Настройка логирования с актуальным файлом на сегодня"""
    log_dir = Path(__file__).parent.parent / "data" / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"fetching_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Удаляем старые обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Файловый обработчик
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# === API-ключи ===
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_API_SECRET")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# === Инициализация клиента OKX ===
from okx import OkxRestClient
client = OkxRestClient(API_KEY, SECRET_KEY, PASSPHRASE)

# === Подключение к Redis ===
try:
    redis_client = redis.Redis(host='okx-redis', port=6379, db=0, decode_responses=False)
    redis_client.ping()
    logger.info("✅ Успешное подключение к Redis")
except redis.ConnectionError:
    logger.error("❌ Не удалось подключиться к Redis")
    redis_client = None

# === Путь к Parquet ===
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def align_timestamp_to_candle(timestamp_ms: int, bar: str) -> int:
    """Выравнивает timestamp по сетке свечей"""
    bar_to_seconds = {
        "1m": 60, "3m": 180, "5m": 300,
        "15m": 900, "30m": 1800, "1h": 3600
    }
    period = bar_to_seconds.get(bar, 180)
    ts_sec = timestamp_ms // 1000
    aligned_sec = (ts_sec // period) * period
    return aligned_sec * 1000

def save_to_parquet(candles: List[Dict], pair: str, bar: str):
    """Сохраняет свечи в Parquet с обработкой дубликатов"""
    filename = PROCESSED_DIR / f"{pair.lower().replace('-', '_')}_{bar}.parquet"
    
    try:
        # Создаем DataFrame из новых свечей
        new_df = pd.DataFrame(candles)
        
        if filename.exists():
            # Загружаем существующие данные
            existing_df = pd.read_parquet(filename)
            
            # Объединяем и удаляем дубликаты
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df.drop_duplicates(subset=["timestamp_ms"], keep="last")
        else:
            combined_df = new_df
        
        # Сортируем по времени
        combined_df = combined_df.sort_values("timestamp_ms", ascending=True)
        combined_df.to_parquet(filename, index=False)
        logger.info(f"💾 Сохранено {len(combined_df)} свечей в {filename}")
        return combined_df
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения в Parquet: {e}")
        return pd.DataFrame()

def load_from_parquet(pair: str, bar: str) -> List[Dict]:
    """Загружает свечи из Parquet"""
    filename = PROCESSED_DIR / f"{pair.lower().replace('-', '_')}_{bar}.parquet"
    if not filename.exists():
        return []
    try:
        df = pd.read_parquet(filename)
        return df.to_dict("records")
    except Exception as e:
        logger.error(f"❌ Ошибка чтения Parquet: {e}")
        return []

def save_to_redis(pair: str, bar: str, candles: List[Dict]):
    """Обновление данных в Redis"""
    if not redis_client:
        logger.warning("Redis не доступен, пропускаем сохранение")
        return
    
    key = f"{pair}:{bar}"
    try:
        pipe = redis_client.pipeline()
        
        # Добавляем новые свечи
        for candle in candles:
            # Используем timestamp_ms как score
            pipe.zadd(key, {json.dumps(candle): candle["timestamp_ms"]})
        
        # Удаляем устаревшие данные (старше 7 дней)
        seven_days_ago = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
        pipe.zremrangebyscore(key, 0, seven_days_ago)
        
        pipe.execute()
        logger.info(f"🔄 Обновлено Redis: {len(candles)} свечей в {key}")
    except Exception as e:
        logger.error(f"❌ Ошибка обновления Redis: {e}")

def load_from_redis(pair: str, bar: str, start_ms: int = None, end_ms: int = None) -> List[Dict]:
    """Загружает свечи из Redis в диапазоне"""
    if not redis_client:
        return []
    
    key = f"{pair}:{bar}"
    if not redis_client.exists(key):
        return []
    
    try:
        if start_ms is None:
            start_ms = 0
        if end_ms is None:
            end_ms = "+inf"
        
        data = redis_client.zrangebyscore(key, start_ms, end_ms)
        return [json.loads(d) for d in data]
    except Exception as e:
        logger.error(f"❌ Ошибка чтения из Redis: {e}")
        return []

def get_btc_usdt_candles(
    start_dt: datetime,
    end_dt: datetime,
    bar: str = "3m",
    max_limit: int = 100,
    force_fetch: bool = False
) -> Optional[List[Dict]]:
    """
    Загружает свечи BTC/USDT в указанном диапазоне.
    Использует пагинацию по `before` для стабильной загрузки.
    """
    setup_logging()

    def to_utc_ms(dt: datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=MSK_TZ)
        return int(dt.astimezone(UTC_TZ).timestamp() * 1000)

    start_ms = to_utc_ms(start_dt)
    end_ms = to_utc_ms(end_dt) - 1  # Защита от будущего
    end_ms = align_timestamp_to_candle(end_ms, bar)

    if start_ms >= end_ms:
        logger.warning(f"⛔ Неверный диапазон: start_ms >= end_ms ({start_ms} >= {end_ms})")
        return None

    pair = "BTC-USDT"

    # 1. Проверка Redis
    if not force_fetch:
        cached = load_from_redis(pair, bar, start_ms, end_ms)
        if cached:
            logger.info(f"♻️ Загружено из кэша Redis: {len(cached)} свечей")
            return cached

        # 2. Проверка Parquet
        all_candles = load_from_parquet(pair, bar)
        df = pd.DataFrame(all_candles) if all_candles else pd.DataFrame()
        if not df.empty and 'timestamp_ms' in df.columns:
            mask = (df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)
            if mask.any():
                result = df[mask].to_dict("records")
                save_to_redis(pair, bar, result)
                logger.info(f"♻️ Загружено из кэша Parquet: {len(result)} свечей")
                return result

    # 3. Запрос к OKX с пагинацией
    all_new_candles = []
    current_end = end_ms
    attempt = 0
    max_attempts = 100  # Максимум 100 запросов

    try:
        while attempt < max_attempts and current_end > start_ms:
            params = {
                "instId": pair,
                "bar": bar,
                "before": str(current_end),
                "limit": str(max_limit)
            }

            logger.debug(f"📡 Запрос #{attempt+1}: before={current_end}")

            response = client.public.get_history_candlesticks(**params)
            if response.get("code") != "0":
                msg = response.get("msg", "Unknown error")
                logger.error(f"❌ Ошибка API: {msg}")
                break

            candles_data = response.get("data", [])
            if not candles_data:
                logger.info("ℹ️ Нет данных в ответе")
                break

            batch = []
            min_ts = None  # Для обновления курсора

            for c in candles_data:
                ts = int(c[0])
                if ts < start_ms:
                    continue
                dt_utc = datetime.fromtimestamp(ts / 1000, tz=UTC_TZ)
                dt_local = dt_utc.astimezone(MSK_TZ)
                batch.append({
                    "timestamp_ms": ts,
                    "time_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "time_local": dt_local.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume_btc": float(c[5]),
                    "volume_usdt": float(c[6]),
                })
                if min_ts is None or ts < min_ts:
                    min_ts = ts

            if not batch:
                logger.info("ℹ️ Нет новых свечей в диапазоне")
                break

            # Добавляем в начало (от старых к новым)
            batch.reverse()
            all_new_candles = batch + all_new_candles

            # Обновляем курсор: на 1 ms меньше самой старой свечи
            current_end = min_ts - 1
            current_end = align_timestamp_to_candle(current_end, bar)

            attempt += 1

        if all_new_candles:
            # Удаляем дубликаты
            seen = set()
            unique_candles = []
            for c in all_new_candles:
                if c["timestamp_ms"] not in seen:
                    seen.add(c["timestamp_ms"])
                    unique_candles.append(c)

            # Сохраняем
            save_to_parquet(unique_candles, pair, bar)
            save_to_redis(pair, bar, unique_candles)
            logger.info(f"✅ Успешно загружено {len(unique_candles)} свечей")
            return unique_candles

        else:
            logger.warning("⚠️ Данные не получены")

    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке: {str(e)}")

    return None