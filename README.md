# Quantitative Asset Allocation Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financialanalysistool-jasonhuang.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Executive Summary

The **Quantitative Asset Allocation Engine** is a professional-grade financial analysis tool designed for real-time portfolio optimization, risk assessment, and active rebalancing. Bridging the gap between academic theory and practical application, this engine allows users to:
1.  **Forecast** future price action using stochastic simulations.
2.  **Optimize** asset allocation using both Classical Mean-Variance and state-of-the-art **Robust Optimization** techniques.
3.  **Rebalance** portfolios to maintain target allocations with precision.

## Key Features

### 1. Robust Portfolio Optimization
Go beyond standard Markowitz optimization. The engine implements advanced statistical methods to handle noisy market data:
-   **Ledoit-Wolf Shrinkage**: Automatically regularizes the sample covariance matrix to reduce estimation error and prevent extreme weights.
-   **Convex Optimization (CVXPY)**: Solves the global minimum variance problem with precision, ensuring mathematically rigorous results unlike approximate random search methods.
-   **Model Comparison**: Dynamically switch between "Classic" (Sample Covariance) and "Robust" (Ledoit-Wolf) models to observe the impact of shrinkage on allocation.

### 2. Portfolio Rebalancing Assistant
Automate the maintenance of your target portfolio structure:
-   **Automated Trade Calculation**: Instantly computes the exact number of shares to buy or sell to align with your target weights.
-   **Integer Share Constraints**: Real-world logic ensures proposed trades are whole numbers, optimizing cash usage while respecting trading limits.
-   **Drift Analysis**: Visualizes the deviation between your current holdings and target allocation to identify rebalancing opportunities.

### 3. Stochastic Forecasting
-   **Monte Carlo Simulation**: Project future asset prices using Geometric Brownian Motion (GBM).
-   **Risk Metrics**: Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR/Expected Shortfall) at a 95% confidence level.
-   **Probability of Loss**: Quantify the likelihood of an investment finishing below its current price.

### 4. Interactive Analysis
-   **Dynamic Efficient Frontier**: Visualize the risk-return trade-off with interactive Plotly charts.
-   **Real-Time Data**: Seamless integration with Yahoo Finance for up-to-the-minute market data.
-   **Correlation Heatmaps**: Instantly identify highly correlated assets to improve diversification.

## Tech Stack

This project leverages a powerful stack of Python libraries standard in the quantitative finance industry:

-   **Core Logic**: `Python`
-   **Frontend**: `Streamlit`
-   **Optimization**: `CVXPY` (Convex Optimization), `Scipy`
-   **Statistical Learning**: `Scikit-Learn` (Ledoit-Wolf Covariance)
-   **Data Analysis**: `Pandas`, `NumPy`
-   **Data Source**: `yfinance`
-   **Visualization**: `Plotly`

## Quick Start

### Prerequisites
-   Python 3.8+
-   pip

### Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
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

## Methodology

### 1. Robust vs. Classical Optimization

| Feature | Classical (Markowitz) | Robust (Ledoit-Wolf + CVXPY) |
| :--- | :--- | :--- |
| **Covariance Input** | Sample Covariance | Ledoit-Wolf Shrinkage |
| **Sensitivity to Noise** | High (Error maximization) | Low (Regularized) |
| **Optimization Method** | Analytical / Random Search | Convex Optimization (Global Minima) |

### 2. Stochastic Forecasting Strategy
The **Stock Price Forecaster** module employs **Geometric Brownian Motion (GBM)**, the standard continuous-time stochastic process for standard market modeling.
-   **Process**: $dS_t = \mu S_t dt + \sigma S_t dW_t$
-   **Drift ($\mu$)**: Derived from historical log-returns to set the trend component.
-   **Volatility ($\sigma$)**: Unbiased historical standard deviation of returns.
-   **Simulation**: Uses vectorized NumPy operations to generate thousands of path-dependent scenarios, enabling the calculation of tail-risk metrics like **VaR (95%)** and **CVaR**.

### 3. Portfolio Rebalancing Logic
The **Rebalancing Assistant** uses a systematic allocation algorithm closer to real-world execution than theoretical percentage splits.
-   **Total Equity Calculation**: Aggregates current cash + current market value of all holdings.
-   **Target Value Mapping**: `Target Value = Total Equity * Target %`
-   **Integer Constraint**: Calculates absolute shares via floor division (`np.floor`) to ensure trades are executable (cannot buy fractional shares on many platforms).
-   **Cash Optimization**: Prioritizes meeting target weights while preventing negative cash balances.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

**Author**: Jason Huang  
**Focus**: HFT, Quantitative Finance, and Machine Learning.
