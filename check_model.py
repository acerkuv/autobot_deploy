import joblib

path = "./models/btc_short_stacked_v2_model_15m_valid.pkl"

print(f"🔍 Загружаю модель из {path}...")
data = joblib.load(path)

print("✅ Ключи в файле:")
print(data.keys())

if 'xgb_model' in data:
    print("🟢 xgb_model найден")
else:
    print("🔴 xgb_model отсутствует")

if 'meta_model' in data:
    print("🟢 meta_model найден")
else:
    print("🔴 meta_model отсутствует")