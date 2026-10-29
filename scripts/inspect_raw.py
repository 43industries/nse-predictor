from pathlib import Path

import pandas as pd

for p in sorted(Path("data/raw/nse").rglob("NSE_data_all_stocks*.csv")):
    df = pd.read_csv(p, nrows=2)
    print(p.name, "->", df.columns.tolist())
    print(" sample date", df.iloc[0, 0])
