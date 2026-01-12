# Customer Sales Analytics – Pricing Intelligence Pipeline

## Overview

This project demonstrates an **end-to-end pricing analytics workflow**, moving from raw transactional data to **descriptive insights**, **predictive modeling**, and **prescriptive pricing recommendations**.

The goal is to answer:

> _How do pricing and discounts affect demand and revenue — and when should discounts be applied?_

This is a **portfolio-grade project**, designed to reflect real-world decision science rather than toy analytics.

---

## Project Structure

---

## Analytical Stages

### Feature Engineering

- Cleans and normalizes raw data
- Merges customer, product, and sales tables
- Engineers pricing, discount, time, and delivery features

**Output**

- `data/processed/features.csv`
- `data/processed/features.parquet`

---

### Descriptive Analytics

Answers:

- Revenue by category, region, and segment
- Discount vs non-discount performance
- Price deviation from base price

---

### Predictive Modeling

#### Quantity Model

- Target: `quantity`
- Purpose: directional demand sensitivity
- Result: low R² (expected for transactional demand)

#### Revenue Model

- Target: `revenue`
- Stronger performance
- Captures pricing, discount, loyalty, and segment effects

---

### Discount Uplift Simulation

- Simulates counterfactual discount scenarios
- Estimates revenue uplift by category, region, and loyalty tier

---

### Prescriptive Pricing

- Converts uplift estimates into actionable rules
- Identifies **where discounts help — and where they hurt**

  _No-discount is a valid recommendation._

---

## How to Run

```bash
python python/features.py
python python/descriptive.py
python python/modeling_quantity.py
python python/modeling_revenue.py
python python/uplift_sim.py
python python/pricing_recommendations.py

```

