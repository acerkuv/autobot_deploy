# ml/train.py
import joblib
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

def train_model():
    print("��� Обучение ансамбля XGBoost + LightGBM...")
    # Здесь будет загрузка данных, фичи, обучение
    model = VotingClassifier([
        ('xgb', XGBClassifier(n_estimators=100)),
        ('lgb', LGBMClassifier(n_estimators=100))
    ])
    # model.fit(X, y)
    joblib.dump(model, "data/models/btc-usdt_ensemble.pkl")
    print("✅ Модель сохранена")

if __name__ == "__main__":
    train_model()
