# Electricity Price Forecasting

This repository contains the implementation of two models, **XGBoost** and **LSTM**, for forecasting wholesale electricity prices. The models are trained on historical data from the Japan Electric Power Exchange (JEPX) to predict the system price for 48 half-hour intervals on a given day.

## Features

- **XGBoost Model**: A machine learning approach that leverages feature engineering and interpretable predictors to achieve high accuracy.
- **LSTM Model**: A deep learning-based approach that captures complex temporal dependencies in electricity prices.
- **Visualization tools** for price predictions, confidence intervals, and feature importance analysis.

## Overview

The Japanese electricity market experiences significant price volatility due to factors such as:

- Seasonal demand fluctuations.
- The integration of renewable energy.
- Market trading activities.

This project addresses these challenges by building models to predict electricity prices, aiding in:

- Load optimization.
- Cost reduction.
- Risk management.

---

## Data Preparation

### Source

- Historical "system price" data was obtained from the Japan Electric Power Exchange (JEPX).

### Preprocessing

- Missing values were filled using interpolation.
- Outliers were removed using Z-scores.
- Data was normalized to improve model convergence.

### Feature Engineering

- **Lagged price variables** (e.g., `price_lag_1`, `price_lag_48`).
- **Rolling statistics** (e.g., moving averages, standard deviations).
- **Market dynamics features** (e.g., `volume_imbalance`, `trading_volume`).
- **Temporal features** (e.g., hour, weekend flags).

---

## Model Training

### XGBoost

- Trained using structured data with explicit feature engineering.
- Hyperparameters optimized via grid search.
- Performance evaluated using metrics such as $\ R^2 \$, MSE, and MAE.

### LSTM

- Neural network with two LSTM layers, each with 128 neurons.
- Optimized using the Adam optimizer and Mean Squared Error loss function.
- Captures temporal dependencies directly from raw data.

---

## Results

### XGBoost

- $\ R^2 \$: 0.9626
- **Peak Price**: 11.19 JPY/kWh at 17:00.
- **Lowest Price**: 10.08 JPY/kWh at 12:00.

### LSTM

- $\ R^2 \$: 0.9439
- **Peak Price**: 14.93 JPY/kWh at 17:00.
- **Lowest Price**: 10.14 JPY/kWh at 11:00.

Both models demonstrated strong performance, with XGBoost excelling in interpretability and LSTM showing flexibility in handling complex patterns.

---

## github link: 
https://github.com/Long-Brian-Yang/energy_market_finace_da_project
