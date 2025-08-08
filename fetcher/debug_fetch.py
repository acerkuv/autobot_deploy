# fetcher/debug_fetch.py
from shared.okx_api import get_btc_usdt_candles
from datetime import datetime, timedelta

# Загружаем данные за последние 10 дней
end_dt = datetime.now() - timedelta(minutes=1)
start_dt = end_dt - timedelta(days=10)

candles = get_btc_usdt_candles(start_dt, end_dt, bar="3m", max_limit=300)
if candles:
    print(f"✅ Успешно загружено {len(candles)} свечей")
else:
    print("❌ Не удалось загрузить данные")