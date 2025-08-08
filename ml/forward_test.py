# ml/forward_test.py
import time
from datetime import datetime, timedelta
from shared.okx_api import get_btc_usdt_candles
from ml.features import create_features
import joblib
import numpy as np

def forward_test_12h():
    model = joblib.load("data/models/btc_usdt_ensemble.pkl")
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=12)
    predictions = []
    actuals = []
    while datetime.now() < end_time:
        try:
            candles = get_btc_usdt_candles(
                start_dt=datetime.now() - timedelta(hours=1),
                end_dt=datetime.now(),
                bar="3m"
            )
            if not candles:
                time.sleep(60)
                continue
            df = pd.DataFrame(candles)
            df = create_features(df)
            if 'target' not in df.columns or len(df) < 2:
                time.sleep(180)
                continue
            X = df[['return', 'sma_5', 'sma_20', 'spread', 'volume_ma', 'rsi']].dropna()
            if len(X) == 0:
                continue
            pred = model.predict(X.iloc[-1:].values)[0]
            true = df.iloc[-1]['target']
            predictions.append(pred)
            actuals.append(true)
            time.sleep(180)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            time.sleep(60)
    if predictions:
        acc = np.mean(np.array(predictions) == np.array(actuals))
        return acc > 0.52
    return False