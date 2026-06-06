# NIFTY IV Surface Prediction
> Predicting missing implied volatility values across strikes and timestamps for NIFTY 50 options data.

## Problem Statement

An implied volatility (IV) surface describes how IV varies across strike prices and timestamps for NIFTY 50 options. In practice, the surface is rarely complete — missing values arise from illiquid strikes, sparse trading, bid-ask filtering, and market microstructure noise.

The task is to impute missing IV values in a provided dataset containing CE (Call) and PE (Put) options at various strikes, minimising Mean Squared Error (MSE) against the hidden ground-truth values. The challenge is hosted as a Kaggle competition with a public/private leaderboard split of 30%/70%.

## Methodology

The project follows a structured, iterative pipeline across 7 notebooks.

### 1. Exploratory Data Analysis

- Parsed CE/PE option columns and extracted strikes from column names.
- Analysed **missingness** — column-wise, daily, and hourly — confirming that missing values are distributed consistently across time, ruling out systematic data gaps.
- Visualised the **IV smile** (IV vs. moneyness) and observed the classic volatility smile pattern, with IV exploding in deep OTM wings near expiry.
- Tracked **ATM IV** over time and examined the distribution of CE vs PE IVs.
- Key observation: the IV surface is structurally smooth across both the strike and time dimensions, motivating neighbour-based and interpolation features.

### 2. Feature Engineering

Transformed the wide-format dataset to a **long format** with one row per (timestamp, option type, strike) observation, and engineered the following features:

| Feature Group | Features |
|---|---|
| Time | `hour`, `minute`, `day_of_week`, `session_progress`, `days_to_expiry` |
| Moneyness | `moneyness` (K/S), `log_moneyness`, `dist_from_atm`, `dist_from_atm_pct` |
| Structure | `is_ce`, `strike_rank` |
| Cross-sectional neighbours | `iv_neighbor_±1`, `iv_neighbor_±2`, `iv_neighbor_mean`, `wide_iv_neighbor_mean` |
| Temporal lags/rolling | `iv_lag_1`, `iv_roll_mean_5`, `iv_roll_mean_10`, `iv_roll_std_5` |
| Cross-area | `mean_ce_iv`, `mean_pe_iv` per timestamp |

All temporal features use only **past values** (via `.shift(1)`) to prevent lookahead bias.

### 3. Baseline Models

Established interpolation and statistical baselines using a masked holdout (20% of observed values hidden):

| Baseline | Method |
|---|---|
| B1 | Forward fill (per strike, over time) |
| B2 | Rolling mean (5-step) |
| B3 | Neighbour mean (adjacent strikes) |
| B4 | Forward fill + neighbour mean blend |
| B5 | Linear interpolation (cross-sectional) |
| B6 | Cubic spline interpolation |
| B7 | Quadratic interpolation |
| B8 | **PCHIP interpolation** (best baseline) |
| B9 | Akima interpolation |

PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) performed best among baselines, preserving monotonicity and avoiding overshoot on the IV smile.

### 4. ML Models

Used a **2-fold time-series cross-validation** (train on first N days, validate on next 2) to evaluate gradient boosting models:

- **XGBoost** (full features) → feature importance used to prune to the most informative subset
- **XGBoost** (pruned features)
- **LightGBM**
- **CatBoost**

### 5. Deep Learning MLP
Input (12 features)
  → Linear(128) → BatchNorm → GELU → Dropout(0.1)
  → Linear(64)  → BatchNorm → GELU → Dropout(0.1)
  → Linear(32)  → BatchNorm → GELU → Dropout(0.1)
  → Linear(1)

Training details:
- Optimiser: Adam with weight decay (`lr=5e-4`, `wd=1e-5`)
- Scheduler: Cosine Annealing LR
- Gradient clipping (norm = 1.0)
- Early stopping with patience = 40 epochs
- 85/15 internal train/val split within each CV fold

### 6. Cascaded Ensemble

Key insight: expiry day behaviour is fundamentally different from regular trading days. Near-expiry IVs are dominated by time-value decay and wing explosions that make ML predictions unreliable.

Designed a cascaded two-branch predictor:
For each missing value:
  ├─ Expiry day  → Cross-sectional interpolation (Linear or Akima)
  └─ Other days  → MLP prediction

Two variants were evaluated:
- Cascaded (Linear + MLP)
- Cascaded (Akima + MLP) — Akima variant was superior, handling wing behaviour more gracefully.

### 7. Final Submission

The final pipeline combines all learnings into a three-component system:

**Component 1 — Expiry Day Fill (Cross-sectional + PCHIP blend)**
- Cross-sectional pass: linear interpolation in log-moneyness space, with slope-damped extrapolation for deep OTM wings.
- Temporal pass: PCHIP interpolation along the time axis for expiry rows.
- Blend: `(1 - α) × cross-sectional + α × temporal` with `α = 0.15`.
- All predictions clipped to a minimum of `0.005`.

**Component 2 — MLP (Non-expiry days)**
- Full 13-feature IVSurfaceMLP trained on all non-expiry data.
- StandardScaler normalisation; CUDA-enabled if available.

**Component 3 — Fallback**
- Forward-fill + back-fill for any remaining NaNs after the two passes.

Final submission is generated as a Kaggle-compatible CSV.

## Results Summary

> MSE values from notebook outputs. The cascaded ensemble consistently outperformed standalone ML and interpolation approaches across both CV folds.

## Data

The dataset is provided as part of the Finance Club IIT Roorkee Open Projects 2026 competition on Kaggle.

## Tech Stack

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Interpolation | `scipy` (interp1d, PCHIP, Akima) |
| ML | `xgboost`, `lightgbm`, `catboost` |
| Deep Learning | `torch` (PyTorch) |
| Utilities | `sklearn` (StandardScaler, MSE) |