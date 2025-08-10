#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Парсер OKX: 10 000 исторических свечей + живой поток
BTC‑USDT, 3‑минутный интервал
"""

import time
import datetime as dt
import requests
import pandas as pd
from tqdm import tqdm

# -----------------------------------  Константы  -----------------------------------
BASE_URL  = "https://www.okx.com/api/v5/market"
INST_ID   = "BTC-USDT"
BAR       = "3m"                # 3‑минутный интервал
LIMIT     = 1000                # максимум свечей за один запрос
TOTAL     = 10000               # требуемое количество исторических свечей
DELAY     = 0.25                 # пауза между запросами (сек.)

# -----------------------------------  Запрос  --------------------------------------
def fetch_candles(before: int | None = None,
                  after: int | None = None) -> dict:
    """GET‑запрос к эндпоинту /candles."""
    params = {
        "instId": INST_ID,
        "bar": BAR,
        "limit": LIMIT
    }
    if before:
        params["before"] = before
    if after:
        params["after"] = after

    resp = requests.get(f"{BASE_URL}/candles", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

# -----------------------------------  Исторические свечи  --------------------------------------
def fetch_historical() -> pd.DataFrame:
    """Скачиваем ровно TOTAL исторических свечей."""
    candles = []
    before = None

    with tqdm(total=TOTAL, desc="Скачивание исторических свечей") as pbar:
        while len(candles) < TOTAL:
            data = fetch_candles(before)
            if data.get("code") != "0":
                raise RuntimeError(f"API‑ошибка {data.get('code')}: {data.get('msg')}")

            batch = data.get("data", [])
            if not batch:
                print("Исторических данных закончились.")
                break

            candles.extend(batch)
            before = batch[-1][0]           # timestamp последней свечи
            time.sleep(DELAY)
            pbar.update(len(batch))

    candles = candles[:TOTAL]
    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close",
        "volume", "count", "turnover", "takerBuyVol"
    ])

    # **Ключевое** – сначала превращаем строку‑число в int
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -----------------------------------  Живой режим  --------------------------------------
def live_update(df: pd.DataFrame,
                last_ts: int,
                csv_path: str = "BTCUSDT_3m_live.csv",
                pause: float = 15.0) -> None:
    """
    После загрузки исторических свечей начинает поллинг новых.
    `last_ts` – timestamp последней свечи в DataFrame (ms).
    """
    # Инициализируем CSV с заголовком
    df.to_csv(csv_path, index=False)
    print(f"\n> Данные сохранены в {csv_path}")
    print("Начинаю живой режим. Прерывание – Ctrl+C\n")

    while True:
        try:
            data = fetch_candles(after=last_ts)
            if data.get("code") != "0":
                raise RuntimeError(f"API‑ошибка {data.get('code')}: {data.get('msg')}")

            batch = data.get("data", [])
            if not batch:
                # Нет новых свечей – подождём
                time.sleep(pause)
                continue

            new_df = pd.DataFrame(batch, columns=[
                "timestamp", "open", "high", "low", "close",
                "volume", "count", "turnover", "takerBuyVol"
            ])
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"].astype(int), unit="ms")

            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(csv_path, index=False)

            # Обновляем last_ts
            last_ts = new_df["timestamp"].iloc[-1].astype('int64') // 10**6
            print(f"✅ Добавлено {len(batch)} новых свечей. "
                  f"Текущий таймстамп: {dt.datetime.fromtimestamp(last_ts/1000)}")

            time.sleep(pause)

        except KeyboardInterrupt:
            print("\n> Прерывание пользователем. Выход.")
            break
        except Exception as exc:
            print(f"\n> Ошибка: {exc}. Попытка повторить через 30 сек.")
            time.sleep(30)

# -----------------------------------  Точка входа  --------------------------------------
if __name__ == "__main__":
    hist_df = fetch_historical()
    # последняя свеча – миллисекунды
    last_timestamp_ms = int(hist_df["timestamp"].iloc[-1].astype('int64') // 10**6)
    live_update(hist_df, last_timestamp_ms)
