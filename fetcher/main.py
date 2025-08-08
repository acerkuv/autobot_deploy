# fetcher/main.py
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from shared.okx_api import get_btc_usdt_candles

# Загружаем .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ .env загружен из {env_path}")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    # Рассчитываем временной диапазон
    end_dt = datetime.now() - timedelta(minutes=5)  # Задержка для закрытия свечей
    start_dt = end_dt - timedelta(hours=1)  # Берем больший диапазон для надежности
    
    # Получаем данные
    candles = get_btc_usdt_candles(
        start_dt=start_dt,
        end_dt=end_dt,
        bar="3m",
        max_limit=300,
        force_fetch=False
    )
    
    if candles:
        # Находим самую свежую свечу
        latest_candle = max(candles, key=lambda x: x["timestamp_ms"])
        logging.info(f"✅ Успешно получены данные. Последняя свеча: {latest_candle['time_local']}")
    else:
        logging.warning("⚠️ Данные не получены")