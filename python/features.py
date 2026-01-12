# features.py
import pandas as pd
from pathlib import Path


def normalize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def to_numeric_clean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
                .replace({"": None, "nan": None, "None": None})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parents[1]
    RAW_DIR = BASE_DIR / "data" / "raw"
    OUT_DIR = BASE_DIR / "data" / "processed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    customers = pd.read_csv(RAW_DIR / "customer_info.csv")
    products = pd.read_csv(RAW_DIR / "product_info.csv")
    sales = pd.read_csv(RAW_DIR / "sales_data.csv")

    # -----------------------------
    # Dates
    # -----------------------------
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], dayfirst=True, errors="coerce")
    products["launch_date"] = pd.to_datetime(products["launch_date"], dayfirst=True, errors="coerce")
    sales["order_date"] = pd.to_datetime(sales["order_date"], dayfirst=True, errors="coerce")

    # -----------------------------
    # Normalize text
    # -----------------------------
    for col in ["gender", "region", "loyalty_tier"]:
        customers[col] = normalize_text(customers[col])

    for col in ["category", "supplier_code"]:
        products[col] = normalize_text(products[col])

    for col in ["delivery_status", "payment_method", "region"]:
        sales[col] = normalize_text(sales[col])

    # -----------------------------
    # Numeric cleanup
    # -----------------------------
    sales = to_numeric_clean(sales, ["quantity", "unit_price", "discount_applied"])
    products = to_numeric_clean(products, ["base_price"])

    sales["discount_applied"] = sales["discount_applied"].fillna(0)

    # SAFE integer conversion (NO crash)
    sales["quantity"] = sales["quantity"].round().astype("Int64")

    # Drop rows that cannot produce revenue
    sales = sales.dropna(subset=["quantity", "unit_price"])

    # -----------------------------
    # Merge
    # -----------------------------
    df = sales.merge(products, on="product_id", how="left", suffixes=("", "_prod"))
    df = df.merge(customers, on="customer_id", how="left", suffixes=("", "_cust"))

    # -----------------------------
    # FORCE region resolution
    # -----------------------------
    # Explicitly handle known bad state
    if "region_x" in df.columns or "region_y" in df.columns:
        df["region"] = df.get("region_x")

        if "region_y" in df.columns:
            df["region"] = df["region"].fillna(df["region_y"])

    # If still missing, hard fail
    if "region" not in df.columns:
        raise RuntimeError("Region column could not be resolved")

    # Drop ALL region variants except canonical one
    for c in list(df.columns):
        if c.startswith("region_"):
            df.drop(columns=c, inplace=True)

    # -----------------------------
    # Feature engineering
    # -----------------------------
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["price_diff"] = df["unit_price"] - df["base_price"]
    df["discount_flag"] = df["discount_applied"] > 0
    df["discount_pct_vs_base"] = df["discount_applied"] / df["base_price"].replace(0, pd.NA)

    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_dow"] = df["order_date"].dt.dayofweek

    df["is_delivered"] = df["delivery_status"] == "delivered"
    df["is_delayed"] = df["delivery_status"] == "delayed"
    df["is_cancelled"] = df["delivery_status"] == "cancelled"

    # Test of region execution
    print("FINAL COLUMNS CONTAINING REGION:", [c for c in df.columns if "region" in c])

    # -----------------------------
    # Save
    # -----------------------------
    df.to_csv(OUT_DIR / "features.csv", index=False)
    df.to_parquet(OUT_DIR / "features.parquet", index=False)

    print("✅ Feature pipeline completed successfully")
    print("Final shape:", df.shape)
    print(df.head())

if __name__ == "__main__":
    main()




