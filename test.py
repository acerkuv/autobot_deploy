# test.py
import pandas as pd

import pandas as pd
df = pd.read_csv("BTCUSDT_3m_10000_candles.csv")
print(df.head)
print(len(set(df)))