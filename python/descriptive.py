# descriptive.py
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0, None) else np.nan


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data" / "processed"
    INSIGHT_DIR = BASE_DIR / "insight"
    OUT_DIR = INSIGHT_DIR / "descriptive"
    FIG_DIR = INSIGHT_DIR / "figures"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load features
    # -----------------------------
    fp_parquet = DATA_DIR / "features.parquet"
    fp_csv = DATA_DIR / "features.csv"

    if fp_parquet.exists():
        df = pd.read_parquet(fp_parquet)
    elif fp_csv.exists():
        df = pd.read_csv(fp_csv)
    else:
        raise FileNotFoundError(
            f"Could not find features.parquet or features.csv in {DATA_DIR}"
        )

    # Basic sanity
    required_cols = [
        "revenue", "unit_price", "base_price", "discount_applied",
        "category", "region", "delivery_status"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in features dataset: {missing}")

    # Ensure numeric
    for c in ["revenue", "unit_price", "base_price", "discount_applied"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Feature proxies (keep consistent with your features.py)
    df["discount_flag"] = df["discount_applied"].fillna(0) > 0
    df["price_diff"] = df["unit_price"] - df["base_price"]          # margin proxy / deviation
    df["discount_pct_vs_base"] = df["discount_applied"] / df["base_price"].replace(0, np.nan)

    # -----------------------------
    # KPI: overall
    # -----------------------------
    total_revenue = df["revenue"].sum(skipna=True)
    total_orders = len(df)
    avg_order_revenue = df["revenue"].mean(skipna=True)

    discount_rate = df["discount_flag"].mean()  # fraction
    avg_discount_value = df.loc[df["discount_flag"], "discount_applied"].mean(skipna=True)
    avg_discount_pct = df.loc[df["discount_flag"], "discount_pct_vs_base"].mean(skipna=True)

    avg_price_diff = df["price_diff"].mean(skipna=True)

    kpi = pd.DataFrame(
        {
            "metric": [
                "total_orders",
                "total_revenue",
                "avg_order_revenue",
                "discount_usage_rate",
                "avg_discount_value_when_discounted",
                "avg_discount_pct_vs_base_when_discounted",
                "avg_price_diff_unit_minus_base",
            ],
            "value": [
                total_orders,
                round(total_revenue, 2),
                round(avg_order_revenue, 2),
                round(discount_rate, 4),
                round(avg_discount_value, 2) if pd.notna(avg_discount_value) else np.nan,
                round(avg_discount_pct, 4) if pd.notna(avg_discount_pct) else np.nan,
                round(avg_price_diff, 2) if pd.notna(avg_price_diff) else np.nan,
            ],
        }
    )
    kpi.to_csv(OUT_DIR / "kpi_overall.csv", index=False)

    # -----------------------------
    # Revenue by category / region
    # -----------------------------
    rev_by_cat = (
        df.groupby("category", dropna=False)["revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    rev_by_cat.to_csv(OUT_DIR / "revenue_by_category.csv", index=False)

    rev_by_region = (
        df.groupby("region", dropna=False)["revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    rev_by_region.to_csv(OUT_DIR / "revenue_by_region.csv", index=False)

    # -----------------------------
    # Discount usage by category
    # -----------------------------
    disc_by_cat = (
        df.groupby("category", dropna=False)
        .agg(
            orders=("discount_flag", "size"),
            discounted_orders=("discount_flag", "sum"),
            avg_discount_value=("discount_applied", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
    )
    disc_by_cat["discount_rate"] = disc_by_cat.apply(
        lambda r: _safe_div(r["discounted_orders"], r["orders"]),
        axis=1,
    )
    disc_by_cat = disc_by_cat.sort_values("discount_rate", ascending=False)
    disc_by_cat.to_csv(OUT_DIR / "discount_usage_by_category.csv", index=False)

    # -----------------------------
    # Discount vs non-discount revenue
    # -----------------------------
    disc_vs = (
        df.groupby("discount_flag", dropna=False)
        .agg(
            orders=("revenue", "size"),
            total_revenue=("revenue", "sum"),
            avg_revenue=("revenue", "mean"),
            avg_price_diff=("price_diff", "mean"),
        )
        .reset_index()
    )
    disc_vs["discount_flag"] = disc_vs["discount_flag"].map({True: "discounted", False: "not_discounted"})
    disc_vs.to_csv(OUT_DIR / "discount_vs_no_discount.csv", index=False)

    # -----------------------------
    # Delivery impact on revenue
    # -----------------------------
    delivery_impact = (
        df.groupby("delivery_status", dropna=False)
        .agg(
            orders=("revenue", "size"),
            total_revenue=("revenue", "sum"),
            avg_revenue=("revenue", "mean"),
            discount_rate=("discount_flag", "mean"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    delivery_impact.to_csv(OUT_DIR / "delivery_impact.csv", index=False)

    # -----------------------------
    # Charts (simple + readable)
    # -----------------------------
    def save_bar(series_df: pd.DataFrame, x: str, y: str, title: str, outpath: Path, top_n: int = 12) -> None:
        plot_df = series_df.head(top_n).copy()
        plt.figure()
        plt.bar(plot_df[x].astype(str), plot_df[y].astype(float))
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()

    save_bar(rev_by_cat, "category", "revenue", "Revenue by Category (Top 12)", FIG_DIR / "revenue_by_category.png")
    save_bar(rev_by_region, "region", "revenue", "Revenue by Region", FIG_DIR / "revenue_by_region.png")

    disc_rate_plot = disc_by_cat[["category", "discount_rate"]].reset_index(drop=True)
    save_bar(disc_rate_plot, "category", "discount_rate", "Discount Usage Rate by Category (Top 12)", FIG_DIR / "discount_rate_by_category.png")

    # Discount vs not discounted revenue chart
    plt.figure()
    plt.bar(disc_vs["discount_flag"].astype(str), disc_vs["total_revenue"].astype(float))
    plt.title("Total Revenue: Discounted vs Not Discounted")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "discount_vs_no_discount_revenue.png")
    plt.close()

    save_bar(delivery_impact, "delivery_status", "total_revenue", "Total Revenue by Delivery Status", FIG_DIR / "delivery_status_revenue.png", top_n=20)

    print("✅ Descriptive pricing analytics completed successfully")
    print(f"Saved tables to: {OUT_DIR}")
    print(f"Saved figures to: {FIG_DIR}")
    print("\nTop revenue categories:")
    print(rev_by_cat.head(5))
    print("\nDiscount vs Non-discount summary:")
    print(disc_vs)


if __name__ == "__main__":
    main()
