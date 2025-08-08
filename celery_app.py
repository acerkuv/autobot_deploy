# celery.py
from celery import Celery
from tasks import retrain_model_task

app = Celery(
    'tasks',
    broker='redis://okx-redis:6379/0',      # ✅ Redis как брокер
    backend='redis://okx-redis:6379/0'      # ✅ Redis как backend
)

app.autodiscover_tasks(['tasks'])

@app.task
def update_btc_data():
    from shared.okx_api import get_btc_usdt_candles
    from datetime import datetime, timedelta
    end_dt = datetime.now() - timedelta(seconds=60)
    start_dt = end_dt - timedelta(minutes=10)
    get_btc_usdt_candles(start_dt, end_dt, bar="3m", max_limit=100)

@app.task
def retrain_model_task():
    from ml.retrain import retrain_model
    import time
    max_retries = 6
    for attempt in range(max_retries):
        print(f"🔁 Попытка {attempt + 1} переобучить модель...")
        if retrain_model():
            print("🎉 Модель успешно обновлена!")
            return True
        if attempt < max_retries - 1:
            print("⏳ Ждём 1 час перед следующей попыткой...")
            time.sleep(3600)
    print("🚨 Все попытки исчерпаны.")
    return False