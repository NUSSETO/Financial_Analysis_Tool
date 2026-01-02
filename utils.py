
import numpy as np
import pandas as pd
import scipy.optimize as sco
import streamlit as st
import yfinance as yf
import cvxpy as cp
from sklearn.covariance import LedoitWolf

# Configuration Constants (Moved from Stock_App.py as needed or defaults)
CACHE_TTL_SECONDS = 3600
VAR_CONFIDENCE_LEVEL = 0.05
MIN_VOLATILITY_FOR_SHARPE = 1e-10
MAX_lines_TO_PLOT = 50

# ==========================================
# Helper Functions (Moved from Stock_App.py)
# ==========================================

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_stock_data(tickers, period):
    """
    Fetches historical stock data from Yahoo Finance for a single ticker or a list of tickers.
    """
    try:
        # Standardize to list for consistent processing
        ticker_list = [tickers] if isinstance(tickers, str) else tickers
        
        # Use yf.download() with ignore_tz=True to avoid timezone issues
        data = yf.download(ticker_list, period=period, ignore_tz=True, progress=False)
        
        if data is None or data.empty:
            return None
        
        return data
            
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_stock_info(ticker):
    """
    Fetches stock information (company name, etc.) from Yahoo Finance.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info if info else None
    except Exception as e:
        return None

def extract_price_data(raw_data, prefer_adj_close=True):
    """
    Extracts price data from raw Yahoo Finance data, handling both single and multi-ticker formats.
    """
    if raw_data is None or raw_data.empty:
        return None
    
    # Handle MultiIndex columns (from batch downloads)
    if isinstance(raw_data.columns, pd.MultiIndex):
        price_col = 'Adj Close' if prefer_adj_close else 'Close'
        if price_col in raw_data.columns.get_level_values(0):
            data = raw_data[price_col]
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(1)
        else:
            price_col = 'Close'
            if price_col in raw_data.columns.get_level_values(0):
                data = raw_data[price_col]
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(1)
            else:
                return None
    else:
        price_col = 'Adj Close' if (prefer_adj_close and 'Adj Close' in raw_data.columns) else 'Close'
        if price_col in raw_data.columns:
            data = raw_data[[price_col]]
        else:
            return None
    
    data = data.dropna(axis=1, how='all')
    return data if not data.empty else None


# ==========================================
# Core Logic Functions
# ==========================================

def run_monte_carlo_simulation(last_price, log_returns, time_horizon, simulations):
    """
    Runs Monte Carlo simulation for stock price forecasting.
    
    Args:
        last_price (float): The most recent closing price.
        log_returns (pd.Series): Historical log returns of the stock.
        time_horizon (int): Number of trading days to simulate.
        simulations (int): Number of simulation scenarios.
        
    Returns:
        dict: A dictionary containing simulation results and metrics.
    """
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    # Vectorized Monte Carlo Simulation (GBM)
    shocks = np.random.normal(0, 1, (time_horizon, simulations))
    drift = mu - 0.5 * sigma**2
    daily_returns_sim = np.exp(drift + sigma * shocks)
    
    # Aggregate into Price Paths
    price_paths = np.vstack([np.ones((1, simulations)), daily_returns_sim])
    price_paths = last_price * price_paths.cumprod(axis=0)
    
    # Compute Metrics
    end_prices = price_paths[-1, :]
    expected_price = float(np.mean(end_prices))
    median_price = float(np.median(end_prices))
    worst_case = float(np.percentile(end_prices, VAR_CONFIDENCE_LEVEL * 100))
    
    tail = end_prices[end_prices <= worst_case]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else worst_case
    prob_loss = float(np.mean(end_prices < last_price))
    
    # Optimize data for visualization
    columns_to_store = min(simulations, MAX_lines_TO_PLOT)
    worst_scenario_idx = int(np.argmin(np.abs(end_prices - worst_case)))
    columns_indices = list(range(columns_to_store))
    
    if worst_scenario_idx not in columns_indices and worst_scenario_idx < simulations:
        columns_indices[-1] = worst_scenario_idx
    
    mean_path_full = np.mean(price_paths, axis=1)
    subset_data = np.column_stack([price_paths[:, columns_indices], mean_path_full])
    subset_columns = [f"Sim_{i}" for i in columns_indices] + ['Mean']
    
    simulation_df = pd.DataFrame(subset_data, columns=subset_columns, index=range(len(price_paths)))
    
    return {
        'simulation_df': simulation_df,
        'expected_price': expected_price,
        'median_price': median_price,
        'worst_case': worst_case,
        'cvar_95': cvar_95,
        'prob_loss': prob_loss
    }

def optimize_portfolio(price_data, risk_free_rate, num_portfolios):
    """
    Performs portfolio optimization using Modern Portfolio Theory.
    
    Args:
        price_data (pd.DataFrame): DataFrame containing price history for assets.
        risk_free_rate (float): The risk-free rate (decimal).
        num_portfolios (int): Number of random portfolios to generate.
        
    Returns:
        dict: Optimization results.
    """
    returns = price_data.pct_change()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Scipy Optimization
    def portfolio_performance(weights, mean_returns, cov_matrix):
        p_ret = np.sum(mean_returns * weights)
        p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return p_ret, p_std

    def neg_sharpe(weights, mean_returns, cov_matrix, rf_rate):
        p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
        # Avoid division by zero
        if p_std < MIN_VOLATILITY_FOR_SHARPE:
            p_std = MIN_VOLATILITY_FOR_SHARPE
        return - (p_ret - rf_rate) / p_std

    num_assets = len(price_data.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    opt_results = sco.minimize(neg_sharpe, init_guess,
                               args=(mean_returns, cov_matrix, risk_free_rate),
                               method='SLSQP',
                               bounds=bounds,
                               constraints=constraints)

    opt_weights = opt_results.x
    opt_ret, opt_std = portfolio_performance(opt_weights, mean_returns, cov_matrix)

    # Simulation
    weights = np.random.random((num_portfolios, num_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]
    
    sim_returns = np.dot(weights, mean_returns)
    sim_variances = np.einsum('ij,jk,ik->i', weights, cov_matrix, weights)
    sim_stds = np.sqrt(sim_variances)
    
    sim_stds_safe = np.where(sim_stds > MIN_VOLATILITY_FOR_SHARPE, sim_stds, MIN_VOLATILITY_FOR_SHARPE)
    sim_sharpes = (sim_returns - risk_free_rate) / sim_stds_safe
    
    results = np.vstack([sim_returns, sim_stds, sim_sharpes])
    
    return {
        'results': results,
        'opt_weights': opt_weights,
        'opt_ret': opt_ret,
        'opt_std': opt_std,
        'returns': returns,  # Needed for correlation matrix
        'tickers': price_data.columns
    }

def optimize_portfolio_robust(returns, risk_free_rate, num_portfolios):
    """
    Performs robust portfolio optimization using Ledoit-Wolf shrinkage and Convex Optimization.
    Also generates random portfolios for visualization using the robust covariance matrix.
    
    Args:
        returns (pd.DataFrame): Daily returns of the assets.
        risk_free_rate (float): The risk-free rate (decimal).
        num_portfolios (int): Number of random portfolios to generate for visualization.
        
    Returns:
        dict: Optimization results containing weights, metrics, and simulation data.
    """
    if returns is None or returns.empty:
        return None

    # 1. Annualize Parameters
    # Clean data: drop rows with NaNs to ensure covariance estimation works
    clean_returns = returns.dropna()
    
    # Simple Mean Returns (Annualized)
    mu = clean_returns.mean().values * 252
    
    # Robust Covariance Estimation (Ledoit-Wolf)
    # Fit on daily returns, then annualize
    lw = LedoitWolf()
    lw.fit(clean_returns)
    Sigma = lw.covariance_ * 252
    
    n_assets = len(mu)
    
    # 2. Define Optimization Problem with CVXPY
    # Variables: weights w
    w = cp.Variable(n_assets)
    
    # Objective: Minimize Variance (w.T @ Sigma @ w) -> equivalent to minimizing quad_form
    risk = cp.quad_form(w, Sigma)
    objective = cp.Minimize(risk)
    
    # Constraints:
    # 1. Sum of weights = 1
    # 2. Weights >= 0 (Long only)
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # 3. Solve
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
    except cp.SolverError:
        # Fallback if solver fails
        return None
        
    if w.value is None:
        return None
        
    # 4. Extract Results
    opt_weights = w.value
    
    # Clean small weights
    opt_weights[opt_weights < 1e-5] = 0
    opt_weights /= opt_weights.sum() # Renormalize
    
    # Calculate Expected Metrics for Optimal Portfolio
    opt_ret = np.dot(opt_weights, mu)
    opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(Sigma, opt_weights)))
    # opt_sharpe = opt_ret / opt_vol if opt_vol > MIN_VOLATILITY_FOR_SHARPE else 0.0 # Not strictly needed for return dict but good to have

    # 5. Generate Random Portfolios for Visualization (using Robust Sigma)
    # This ensures the scatter plot aligns with the robust assumptions
    weights_sim = np.random.random((num_portfolios, n_assets))
    weights_sim /= np.sum(weights_sim, axis=1)[:, np.newaxis]
    
    sim_returns = np.dot(weights_sim, mu)
    sim_variances = np.einsum('ij,jk,ik->i', weights_sim, Sigma, weights_sim)
    sim_stds = np.sqrt(sim_variances)
    
    sim_stds_safe = np.where(sim_stds > MIN_VOLATILITY_FOR_SHARPE, sim_stds, MIN_VOLATILITY_FOR_SHARPE)
    sim_sharpes = (sim_returns - risk_free_rate) / sim_stds_safe
    
    results = np.vstack([sim_returns, sim_stds, sim_sharpes])
    
    return {
        'results': results,
        'opt_weights': opt_weights,
        'opt_ret': opt_ret,
        'opt_std': opt_vol,
        'returns': returns,
        'tickers': returns.columns
    }

def calculate_rebalancing_plan(current_cash, valid_rows, current_prices_dict):
    """
    Calculates the rebalancing plan.
    
    Args:
        current_cash (float): The current cash balance.
        valid_rows (pd.DataFrame): DataFrame with Ticker, Shares, Target (%).
        current_prices_dict (dict): Dictionary of {ticker: price}.
        
    Returns:
        dict: Rebalancing results.
    """
    results = []
    total_equity = current_cash
    
    # Calculate Total Portfolio Value
    for _, row in valid_rows.iterrows():
        ticker = row['Ticker'].upper()
        shares = row['Shares']
        price = current_prices_dict.get(ticker, 0.0)
        total_equity += shares * price
        
    if total_equity <= 0:
        return {'error': "Total portfolio value is zero or negative."}

    projected_cash = total_equity
    
    for _, row in valid_rows.iterrows():
        ticker = row['Ticker'].upper()
        current_shares = row['Shares']
        target_pct = row['Target (%)'] / 100.0
        price = current_prices_dict.get(ticker, 0.0)
        
        if price > 0:
            target_value = total_equity * target_pct
            new_shares = int(np.floor(target_value / price))
            trade_shares = new_shares - current_shares
            final_value = new_shares * price
            projected_cash -= final_value
            actual_weight = (final_value / total_equity) * 100
            
            results.append({
                "Ticker": ticker,
                "New Shares": new_shares,
                "Trade (+/-)": trade_shares,
                "Value ($)": final_value,
                "Actual %": actual_weight
            })
            
    res_df = pd.DataFrame(results)
    
    return {
        'results_df': res_df,
        'total_equity': total_equity,
        'projected_cash': projected_cash
    }
