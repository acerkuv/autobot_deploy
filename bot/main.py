# bot/main.py
import os
import time
import joblib
import pandas as pd
from datetime import datetime, timedelta
from shared.okx_api import get_btc_usdt_candles
from bot.risk import calculate_position_size, get_tp_sl_levels
from okx import OkxRestClient

# Загружаем переменные из .env
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_API_SECRET")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# ✅ Правильное создание клиента — позиционные аргументы
client = OkxRestClient(API_KEY, SECRET_KEY, PASSPHRASE)

MODEL_FILE = "data/models/btc_usdt_ensemble.pkl"

def should_trade():
    if not os.path.exists(MODEL_FILE):
        return False
    mod_time = datetime.fromtimestamp(os.path.getmtime(MODEL_FILE))
    if datetime.now() - mod_time > timedelta(hours=4):
        return False
    return True

def trading_loop():
    print("🚀 LONG-бот запущен")
    while True:
        if not should_trade():
            print("🛑 Модель не валидна. Ожидание...")
            time.sleep(300)
            continue
        try:
            end_dt = datetime.now() - timedelta(seconds=60)
            start_dt = end_dt - timedelta(hours=1)
            candles = get_btc_usdt_candles(start_dt, end_dt, bar="3m")
            if not candles:
                time.sleep(60)
                continue
            df = pd.DataFrame(candles)
            df['close'] = df['close'].astype(float)
            df['return'] = df['close'].pct_change()
            df['sma_5'] = df['close'].rolling(5).mean()
            df['rsi'] = (df['close'].diff(1).clip(lower=0).rolling(14).mean()) / \
                        (df['close'].diff(1).abs().rolling(14).mean()) * 100
            latest = df[['return', 'sma_5', 'rsi']].dropna().iloc[-1:].values
            model = joblib.load(MODEL_FILE)
            prob = model.predict_proba(latest)[0][1]
            price = df.iloc[-1]['close']
            print(f"📊 [LONG] Вероятность роста: {prob:.2f}, Цена: {price:.2f}")
            if prob > 0.6:
                size = calculate_position_size(1000, price)
                tp_sl = get_tp_sl_levels(price, strategy="moderate")
                print(f"📈 [BUY] Покупаем {size:.6f} BTC по {price:.2f}")
                # Раскомментируй, когда будешь готов торговать
                # client.trade.place_order(...)
            time.sleep(180)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            time.sleep(60)

if __name__ == "__main__":
    trading_loop()