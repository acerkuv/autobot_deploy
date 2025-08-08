# ml/retrain.py
import joblib
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from ml.features import create_features
from ml.backtest import backtest_model
from ml.forward_test import forward_test_12h

MODEL_FILE = "data/models/btc_usdt_ensemble.pkl"

def retrain_model():
    df = pd.read_parquet("data/processed/btc_usdt_3m.parquet")
    df = create_features(df)
    X = df[['return', 'sma_5', 'sma_20', 'spread', 'volume_ma', 'rsi']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    xgb = XGBClassifier(n_estimators=100)
    lgb = LGBMClassifier(n_estimators=100)
    model = VotingClassifier([('xgb', xgb), ('lgb', lgb)], voting='soft')
    model.fit(X_train, y_train)
    if not forward_test_12h():
        return False
    for days in [1, 30, 60]:
        result = backtest_model(model, df, days)
        if not result or result['accuracy'] < 0.55 or result['precision'] < 0.5:
            return False
    joblib.dump(model, MODEL_FILE)
    return True