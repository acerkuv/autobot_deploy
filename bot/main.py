# bot/main.py
import pandas as pd
import os

def is_model_valid(log_path="./logs/model_training_log.csv", min_auc=0.65):
    if not os.path.exists(log_path):
        print("❌ Лог обучения не найден. Пропуск торговли.")
        return False
    log_df = pd.read_csv(log_path)
    latest = log_df.iloc[-1]
    if latest['status'] == 'valid' and latest['auc'] > min_auc:
        print(f"✅ Модель валидна: AUC={latest['auc']:.3f}")
        return True
    else:
        print(f"🛑 Модель не прошла валидацию: AUC={latest['auc']:.3f}")
        return False

# Пример использования
if is_model_valid():
    print("✅ Разрешаем торговлю")
else:
    print("🛑 Торговля приостановлена")