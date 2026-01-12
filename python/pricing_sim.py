# pricing_sim.py
from __future__ import annotations

import pandas as pd
from pathlib import Path


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parents[1]
    IN_DIR = BASE_DIR / "insight" / "predictive"
    OUT_DIR = BASE_DIR / "insight" / "prescriptive"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_DIR / "discount_uplift_by_segment.csv")

    # -----------------------------
    # Business constraints
    # -----------------------------
    MIN_UPLIFT_PCT = 0.05     # at least +5% revenue
    MIN_SAMPLES = 30         # reliability threshold
    MAX_DISCOUNT = 0.10      # guardrail (no over-discounting)

    recs = df[
        (df["uplift_pct"] >= MIN_UPLIFT_PCT) &
        (df["n"] >= MIN_SAMPLES) &
        (df["discount_rate"] <= MAX_DISCOUNT)
    ].copy()

    # Rank recommendations
    recs = recs.sort_values(
        ["uplift_pct", "uplift_abs"],
        ascending=False
    )

    # Select best discount per segment
    best = (
        recs.groupby(["category", "region", "loyalty_tier"])
        .head(1)
        .reset_index(drop=True)
    )

    out_fp = OUT_DIR / "pricing_recommendations.csv"
    best.to_csv(out_fp, index=False)

    print(" Prescriptive pricing recommendations generated")
    print(f"Saved to: {out_fp}\n")

    print("Top pricing recommendations:")
    print(
        best[
            [
                "category",
                "region",
                "loyalty_tier",
                "discount_rate",
                "uplift_pct",
                "uplift_abs",
                "n",
            ]
        ]
        .head(15)
    )


if __name__ == "__main__":
    main()
