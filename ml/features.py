# ml/features.py
import pandas as pd

def create_features(df):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['spread'] = df['high'] - df['low']
    df['volume_ma'] = df['volume_usdt'].rolling(5).mean()
    df['rsi'] = (df['close'].diff(1).clip(lower=0).rolling(14).mean()) / \
                (df['close'].diff(1).abs().rolling(14).mean()) * 100
    df['target'] = (df['close'].shift(-12) > df['close']).astype(int)
    return df.dropna()