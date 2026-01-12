# modeling.py
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data" / "processed"
    OUT_DIR = BASE_DIR / "insight" / "predictive"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_parquet(DATA_DIR / "features.parquet")

    # -----------------------------
    # Target (STEP 1)
    # -----------------------------
    TARGET = "quantity"

    # Drop rows with missing target or core price info
    df = df.dropna(subset=[TARGET, "unit_price", "base_price"])

    # -----------------------------
    # Feature selection
    # -----------------------------
    numeric_features = [
        "unit_price",
        "base_price",
        "discount_applied",
        "order_month",
        "order_dow",
    ]

    categorical_features = [
        "category",
        "region",
        "loyalty_tier",
    ]

    X = df[numeric_features + categorical_features]
    y = df[TARGET]

    # -----------------------------
    # Preprocessing (with imputation)
    # -----------------------------
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = LinearRegression()

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # -----------------------------
    # Train / test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_pred = pipeline.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(" Quantity model trained successfully")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    # -----------------------------
    # Coefficients (interpretability)
    # -----------------------------
    feature_names = (
        numeric_features
        + pipeline.named_steps["preprocess"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )

    coefs = pipeline.named_steps["model"].coef_

    coef_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefs,
            }
        )
        .sort_values("coefficient", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    coef_df.to_csv(OUT_DIR / "quantity_model_coefficients.csv", index=False)

    print("\nTop positive drivers of quantity:")
    print(coef_df.head(10))

    print("\nTop negative drivers of quantity:")
    print(coef_df.tail(10))

    print(f"\nSaved coefficients to: {OUT_DIR / 'quantity_model_coefficients.csv'}")


if __name__ == "__main__":
    main()
