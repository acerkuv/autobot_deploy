# celery_app.py
from celery import Celery
from datetime import datetime, timedelta
from ml.retrain import retrain_model
import time
import tasks
from tasks import retrain_model_task

# ✅ Убедись, что Redis доступен по имени сервиса
app = Celery(
    'tasks',
    broker='redis://okx-redis:6379/0',
    backend='redis://okx-redis:6379/0'
)

@app.task
def update_btc_data():
    from shared.okx_api import get_btc_usdt_candles
    now = datetime.now()
    end_dt = now - timedelta(seconds=60)
    start_dt = end_dt - timedelta(minutes=15)
    print(f"🔄 [Celery] Обновление данных: {start_dt} → {end_dt}")
    get_btc_usdt_candles(start_dt, end_dt, bar="3m")

@app.task
def retrain_model_task():
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