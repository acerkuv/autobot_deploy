# test.py
import pandas as pd

df = pd.read_parquet("data/processed/btc_usdt_3m.parquet")

# ✅ Убедимся, что timestamp_ms — в миллисекундах
print("Первые 5 значений timestamp_ms:")
print(df["timestamp_ms"].head())

# ✅ Конвертируем в timestamp
df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

print(f"\nВсего свечей: {len(df)}")
print(f"Первая: {df['timestamp'].min()}")
print(f"Последняя: {df['timestamp'].max()}")