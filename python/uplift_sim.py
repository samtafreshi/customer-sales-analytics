# uplift_sim.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data" / "processed"
    OUT_DIR = BASE_DIR / "insight" / "predictive"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_DIR / "features.parquet")

    # Target: revenue
    df = df.dropna(subset=["revenue", "unit_price", "base_price"])
    df = df[df["revenue"] >= 0].copy()

    numeric_features = ["unit_price", "base_price", "discount_applied", "order_month", "order_dow"]
    categorical_features = ["category", "region", "loyalty_tier"]

    X = df[numeric_features + categorical_features].copy()
    y = df["revenue"].copy()

    # Preprocess
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ])

    # Non-linear model (good baseline for pricing effects)
    model = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.05,
    )

    pipe = Pipeline([("preprocess", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print("✅ Uplift model trained successfully (HGBRegressor)")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    # -----------------------------
    # Discount simulation
    # -----------------------------
    discount_grid = [0.0, 0.05, 0.10, 0.15, 0.20]

    sim_base = X_test.copy()

    # We'll approximate discount_applied from base_price * discount_rate
    # and adjust unit_price as base_price * (1 - discount_rate)
    # This assumes unit_price tracks discounted price.
    rows = []

    for d in discount_grid:
        sim = sim_base.copy()
        sim["discount_applied"] = sim["base_price"] * d
        sim["unit_price"] = sim["base_price"] * (1 - d)

        sim_pred_rev = pipe.predict(sim)

        rows.append(pd.DataFrame({
            "discount_rate": d,
            "pred_revenue": sim_pred_rev,
            "category": sim["category"].values,
            "region": sim["region"].values,
            "loyalty_tier": sim["loyalty_tier"].values,
        }))

    sim_df = pd.concat(rows, ignore_index=True)

    # Baseline = 0% discount
    base = sim_df[sim_df["discount_rate"] == 0.0].copy()
    base = base.rename(columns={"pred_revenue": "pred_revenue_base"}).drop(columns=["discount_rate"])

    merged = sim_df.merge(base, on=["category", "region", "loyalty_tier"], how="left")
    merged["uplift_abs"] = merged["pred_revenue"] - merged["pred_revenue_base"]
    merged["uplift_pct"] = merged["uplift_abs"] / merged["pred_revenue_base"].replace(0, np.nan)

    # Aggregate by segment + discount rate
    uplift_seg = (
        merged.groupby(["category", "region", "loyalty_tier", "discount_rate"])
        .agg(
            avg_pred_rev=("pred_revenue", "mean"),
            avg_base=("pred_revenue_base", "mean"),
            uplift_abs=("uplift_abs", "mean"),
            uplift_pct=("uplift_pct", "mean"),
            n=("pred_revenue", "size"),
        )
        .reset_index()
        .sort_values(["uplift_abs"], ascending=False)
    )

    out_fp = OUT_DIR / "discount_uplift_by_segment.csv"
    uplift_seg.to_csv(out_fp, index=False)

    print(f"\n Saved uplift table to: {out_fp}")
    print("\nTop uplift segments (avg absolute uplift):")
    print(uplift_seg.head(15))


if __name__ == "__main__":
    main()
