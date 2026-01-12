import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"

# Load data
customers = pd.read_csv(DATA_DIR / "customer_info.csv")
products = pd.read_csv(DATA_DIR / "product_info.csv")
sales = pd.read_csv(DATA_DIR / "sales_data.csv")

# Basic cleaning
customers["signup_date"] = pd.to_datetime(customers["signup_date"], dayfirst=True)
products["launch_date"] = pd.to_datetime(products["launch_date"], dayfirst=True)
sales["order_date"] = pd.to_datetime(sales["order_date"], dayfirst=True)

# Normalize text
customers["gender"] = customers["gender"].str.lower()
customers["loyalty_tier"] = customers["loyalty_tier"].str.lower()
sales["delivery_status"] = sales["delivery_status"].str.lower()
sales["payment_method"] = sales["payment_method"].str.lower()

print("Cleaning completed successfully")