# Financial Analysis Tool

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financialanalysistool-jasonhuang.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/NUSSETO/Financial_Analysis_Tool/actions/workflows/python-app.yml/badge.svg)](https://github.com/NUSSETO/Financial_Analysis_Tool/actions/workflows/python-app.yml)

## Executive Summary

The **Financial Analysis Tool** is a professional-grade financial analysis tool designed for portfolio optimization, risk assessment, backtesting, and active rebalancing. Bridging the gap between academic theory and practical application, this engine allows users to:
1.  **Forecast** future price action using stochastic simulations with comprehensive risk metrics.
2.  **Optimize** asset allocation using four models: Classical Mean-Variance, **Robust Optimization**, **Black-Litterman**, and **Risk Parity**.
3.  **Rebalance** portfolios to maintain target allocations with precision.
4.  **Stress Test** portfolios against historical market crises.
5.  **Backtest** portfolio strategies with benchmark comparison.
6.  **Analyze Risk** with rolling metrics and CAPM factor decomposition.

## Key Features

### 1. Stochastic Forecasting
-   **Monte Carlo Simulation**: Project future asset prices using Geometric Brownian Motion (GBM).
-   **Risk Metrics**: Calculate Value at Risk (VaR), Conditional Value at Risk (CVaR/Expected Shortfall), **Maximum Drawdown**, and **Annualized Sharpe Ratio**.
-   **Terminal Price Histogram**: Visualize the full distribution of simulated end-of-period prices with reference lines for current price, expected price, and VaR threshold.
-   **Probability of Loss**: Quantify the likelihood of an investment finishing below its current price.

### 2. Portfolio Optimization (4 Models)
Go beyond standard Markowitz optimization with four distinct models:

| Model | Description |
| :--- | :--- |
| **Classic (Sample Covariance)** | Standard Mean-Variance optimization using sample covariance |
| **Robust (Ledoit-Wolf)** | Ledoit-Wolf shrinkage regularization + CVXPY convex optimization |
| **Black-Litterman** | Blend your personal return views with market equilibrium returns |
| **Risk Parity** | Equal risk contribution from each asset (Bridgewater-style) |

-   **Efficient Frontier Curve**: True frontier computed via constrained optimization (CVXPY), overlaid on the scatter plot.
-   **Allocation Pie Chart**: Interactive donut chart for optimal weight visualization.
-   **Correlation Heatmaps**: Instantly identify highly correlated assets.

### 3. Portfolio Rebalancing Assistant
-   **Automated Trade Calculation**: Computes exact shares to buy/sell to align with target weights.
-   **Integer Share Constraints**: Whole-number trades with optimized cash usage.
-   **Drift Analysis**: Visualizes deviation between current holdings and target allocation.

### 4. Historical Stress Testing
-   **4 Major Crises**: Dot-Com Crash (2000–2002), Global Financial Crisis (2007–2009), COVID-19 Crash (2020), 2022 Bear Market.
-   **Crisis Metrics**: Total return, maximum drawdown, days to trough, 6-month recovery status.
-   **Cumulative Return Charts**: Interactive charts per crisis period.

### 5. Portfolio Backtester
-   **Customizable Strategy**: Pick tickers, weights, date range, and rebalancing frequency (none, monthly, quarterly, annually).
-   **Benchmark Comparison**: Side-by-side cumulative returns vs SPY (or any benchmark).
-   **Key Metrics**: Total Return, CAGR, Max Drawdown, Sharpe Ratio, Win Rate.
-   **Drawdown Chart**: Visualize underwater periods with filled area chart.

### 6. Risk Dashboard
-   **Rolling Metrics**: Rolling annualized volatility, Sharpe ratio, and beta (vs benchmark) with adjustable window size.
-   **CAPM Factor Decomposition**: Regression-based analysis yielding Alpha (annualized), Beta, R², Tracking Error, and Information Ratio.
-   **Educational Explainers**: Built-in tooltips and expandable guides for each metric.

## Tech Stack

-   **Core Logic**: `Python`
-   **Frontend**: `Streamlit`
-   **Optimization**: `CVXPY` (Convex Optimization), `Scipy`
-   **Statistical Learning**: `Scikit-Learn` (Ledoit-Wolf Covariance, Linear Regression)
-   **Data Analysis**: `Pandas`, `NumPy`
-   **Data Source**: `yfinance`
-   **Visualization**: `Plotly`

## Quick Start

### Prerequisites
-   Python 3.10+
-   pip

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/NUSSETO/Financial_Analysis_Tool.git
    cd Financial_Analysis_Tool
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**
    ```bash
    streamlit run Stock_App.py
    ```

    The application will launch automatically in your default web browser at `http://localhost:8501`.

4.  **Run tests**
    ```bash
    python -m pytest tests/ -v
    ```

## Methodology

### 1. Optimization Models

| Feature | Classical | Robust (Ledoit-Wolf) | Black-Litterman | Risk Parity |
| :--- | :--- | :--- | :--- | :--- |
| **Covariance** | Sample | Ledoit-Wolf Shrinkage | Ledoit-Wolf Shrinkage | Ledoit-Wolf Shrinkage |
| **Expected Returns** | Historical Mean | Historical Mean | Equilibrium + User Views | Historical Mean |
| **Objective** | Max Sharpe | Min Variance | Max Utility (adjusted) | Equal Risk Contribution |
| **Sensitivity** | High | Low | Low | Low |

### 2. Stochastic Forecasting Strategy
The **Stock Price Forecaster** module employs **Geometric Brownian Motion (GBM)**, the standard continuous-time stochastic process for standard market modeling.
-   **Process**: $dS_t = \mu S_t dt + \sigma S_t dW_t$
-   **Drift ($\mu$)**: Derived from historical log-returns to set the trend component.
-   **Volatility ($\sigma$)**: Unbiased historical standard deviation of returns.
-   **Simulation**: Uses vectorized NumPy operations to generate thousands of path-dependent scenarios.
-   **Metrics**: VaR (95%), CVaR, Maximum Drawdown (worst peak-to-trough across all paths), and Annualized Sharpe Ratio.

### 3. Black-Litterman Model
Combines market equilibrium with user views using the formula:
-   **Equilibrium returns**: $\pi = \delta \Sigma w_{eq}$ (reverse optimization with risk aversion $\delta = 2.5$)
-   **Posterior**: $E[R] = [(\tau\Sigma)^{-1} + P'\Omega^{-1}P]^{-1} [(\tau\Sigma)^{-1}\pi + P'\Omega^{-1}Q]$
-   View uncertainty ($\Omega$) is proportional to asset variance, scaled by $\tau = 0.05$

### 4. Risk Parity
Solves for weights where each asset contributes equally to total portfolio risk:
-   **Objective**: Minimize $\sum_i (RC_i - \frac{1}{N})^2$ where $RC_i = \frac{w_i (\Sigma w)_i}{w^T \Sigma w}$
-   Uses SLSQP optimization with long-only constraints

### 5. Portfolio Rebalancing Logic
-   **Total Equity Calculation**: Aggregates current cash + current market value of all holdings.
-   **Target Value Mapping**: `Target Value = Total Equity * Target %`
-   **Integer Constraint**: Calculates absolute shares via floor division (`np.floor`).
-   **Cash Optimization**: Prioritizes meeting target weights while preventing negative cash balances.

### 6. CAPM Factor Decomposition
-   **Regression**: $R_p - R_f = \alpha + \beta(R_m - R_f) + \epsilon$
-   Reports: Annualized Alpha, Beta, R², Tracking Error, Information Ratio

## Assumptions

1.  **Data Processing**:
    *   **Source**: Yahoo Finance (`yfinance`).
    *   **Price Type**: Prefers **Adjusted Close** prices to account for splits and dividends; falls back to **Close** prices if unavailable.

2.  **Returns Calculation**:
    *   **Portfolio Optimization**: Uses **Daily Simple Returns** annualized with **252 trading days**.
    *   **Monte Carlo Forecasting**: Uses **Daily Log Returns** consistent with GBM model requirements.

3.  **Optimization Constraints**:
    *   **Long-Only**: Weights must be non-negative ($w_i \ge 0$). Short selling is not permitted.
    *   **Fully Invested**: Weights must sum to exactly 1.0 ($\sum w_i = 1$). No leverage.
    *   **Frictionless**: Transaction costs are not currently modeled.

4.  **Risk Metrics (VaR/CVaR)**:
    *   **Confidence Level**: 95%.
    *   **VaR**: 5th percentile of simulated terminal price distribution.
    *   **CVaR (Expected Shortfall)**: Average of all outcomes below VaR.

5.  **Rebalancing Logic**:
    *   **Integer Constraints**: Whole shares via `floor` rounding.
    *   **Cash Management**: Validates allocations don't exceed 100%; flags negative projected cash. Transaction fees excluded.

## Testing

The project includes **27 unit tests** covering:
-   Price data extraction (4 tests)
-   Monte Carlo simulation metrics (1 test)
-   Portfolio rebalancing (2 tests)
-   Efficient frontier computation (2 tests)
-   Robust optimization (4 tests)
-   Risk parity optimization (1 test)
-   Black-Litterman optimization (1 test)
-   Factor decomposition (2 tests)
-   Rolling risk metrics (2 tests)
-   Edge cases (8 tests)

```bash
python -m pytest tests/ -v
```

## Disclaimer

**Educational Use Only**: This application is intended for **informational and educational purposes only**. It uses historical data and statistical models to demonstrate theoretical concepts.

**Not Financial Advice**: The content, projections, and analysis generated by this tool do **not** constitute financial advice, investment recommendations, or an offer to buy or sell any securities.

**No Guarantees**: All financial systems involve inherent risks. Past performance and historical volatility are not indicative of future results. The authors assume no liability for any financial losses incurred from the use of this software.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

**Author**: Jason Huang  
**Focus**: HFT, Quantitative Finance, and Machine Learning.
