import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # customer_sales_analytics/
DATA_DIR = BASE_DIR / "data" / "raw"

files = [
    "customer_info.csv",
    "product_info.csv",
    "sales_data.csv"
]

for f in files:
    path = DATA_DIR / f
    df = pd.read_csv(path)
    print(f"\n{f}")
    print(df.columns.tolist())

