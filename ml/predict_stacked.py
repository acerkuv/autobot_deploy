# ml/predict_stacked.py
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Загрузка модели
stacked = joblib.load("./models/btc_long_stacked_model.pkl")
lstm_model = load_model("./models/lstm_model.h5")

# Прогноз
def predict(X_seq, X_flat):
    lstm_proba = lstm_model.predict(X_seq).flatten()
    lgb_proba = stacked['lgb_model'].predict_proba(X_flat)[:, 1]
    stacked_proba = stacked['meta_model'].predict_proba(np.column_stack([lstm_proba, lgb_proba]))[:, 1]
    return stacked_proba