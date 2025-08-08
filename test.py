# test.py
import pandas as pd

import pandas as pd
df = pd.read_parquet("data/processed/btc_usdt_3m.parquet")
print(df.head)