# tasks.py
from celery import Celery
from datetime import datetime, timedelta
from celery import shared_task
from models.model import train_model
# from shared.okx_api import get_btc_usdt_candles
import pandas as pd


# Подключаемся к тому же брокеру
app = Celery('tasks', broker='redis://okx-redis:6379/0', backend='redis://okx-redis:6379/0')

@app.task
def update_btc_data():
    """Обновляет последние 3-минутные свечи BTC/USDT"""
    from shared.okx_api import get_btc_usdt_candles

    now = datetime.now()
    end_dt = now - timedelta(seconds=60)  # Чтобы не попасть в незакрытую свечу
    start_dt = end_dt - timedelta(minutes=15)  # Запрашиваем с запасом

    print(f"🔄 [Celery] Обновление данных: {start_dt} → {end_dt}")

    try:
        candles = get_btc_usdt_candles(start_dt, end_dt, bar="3m", max_limit=300)
        if candles:
            print(f"✅ [Celery] Добавлено {len(candles)} новых свечей")
        else:
            print("❌ [Celery] Нет новых данных")
    except Exception as e:
        print(f"❌ [Celery] Ошибка: {e}")


@shared_task
def retrain_model_task():
    print("🔄 Запуск переобучения модели...")

    try:
        # Загружаем данные
        df_3m = pd.read_parquet("data/processed/btc_usdt_3m.parquet")
        df_1h = pd.read_parquet("data/processed/btc_usdt_1h.parquet")

        # Обучаем модель
        model = train_model(df_3m, df_1h)

        # Сохраняем (пример)
        import joblib
        joblib.dump(model, "models/btc_long_model.pkl")

        print("✅ Модель успешно обучена и сохранена")
        return "LONG"  # или вероятность

    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        return "ERROR"

# Расписание
# app.conf.beat_schedule = {
#     'update-btc-data-every-3-minutes': {
#         'task': 'tasks.update_btc_data',
#         'schedule': 180.0,  # Каждые 3 минуты
#     },
# }
# app.conf.timezone = 'UTC'

