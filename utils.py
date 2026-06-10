
import numpy as np
import pandas as pd
import scipy.optimize as sco
import streamlit as st
import yfinance as yf
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression

# Configuration Constants (Moved from Stock_App.py as needed or defaults)
CACHE_TTL_SECONDS = 3600
VAR_CONFIDENCE_LEVEL = 0.05
MIN_VOLATILITY_FOR_SHARPE = 1e-10
MAX_LINES_TO_PLOT = 50

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
    
    # Maximum Drawdown: worst peak-to-trough drop across all simulated paths
    running_max = np.maximum.accumulate(price_paths, axis=0)
    drawdowns = (price_paths - running_max) / running_max  # negative values
    max_drawdown = float(np.min(drawdowns))  # most negative = worst drawdown
    
    # Annualized Sharpe Ratio (from simulated returns, assuming 0 risk-free rate)
    sim_total_returns = (end_prices / last_price) - 1.0
    annualization_factor = 252.0 / time_horizon
    ann_mean = float(np.mean(sim_total_returns) * annualization_factor)
    ann_std = float(np.std(sim_total_returns) * np.sqrt(annualization_factor))
    sharpe_ratio = ann_mean / ann_std if ann_std > MIN_VOLATILITY_FOR_SHARPE else 0.0
    
    # Optimize data for visualization
    columns_to_store = min(simulations, MAX_LINES_TO_PLOT)
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
        'prob_loss': prob_loss,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'end_prices': end_prices
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

def compute_efficient_frontier(mean_returns, cov_matrix, num_points=50):
    """
    Traces the Efficient Frontier by solving minimum-variance portfolios
    at a series of target return levels.
    
    Args:
        mean_returns (np.ndarray): Annualized mean returns per asset.
        cov_matrix (np.ndarray): Annualized covariance matrix.
        num_points (int): Number of points along the frontier.
        
    Returns:
        dict: {'frontier_vols': np.ndarray, 'frontier_rets': np.ndarray}
              or None if optimization fails.
    """
    n_assets = len(mean_returns)
    
    # Determine return range from individual asset returns
    min_ret = float(np.min(mean_returns))
    max_ret = float(np.max(mean_returns))
    
    if np.isclose(min_ret, max_ret):
        return None
    
    target_returns = np.linspace(min_ret, max_ret, num_points)
    frontier_vols = []
    frontier_rets = []
    
    for target in target_returns:
        w = cp.Variable(n_assets)
        risk = cp.quad_form(w, cov_matrix)
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            mean_returns @ w >= target
        ]
        prob = cp.Problem(cp.Minimize(risk), constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            continue
        
        if w.value is not None:
            vol = float(np.sqrt(np.dot(w.value.T, np.dot(cov_matrix, w.value))))
            ret = float(np.dot(w.value, mean_returns))
            frontier_vols.append(vol)
            frontier_rets.append(ret)
    
    if len(frontier_vols) < 2:
        return None
    
    return {
        'frontier_vols': np.array(frontier_vols),
        'frontier_rets': np.array(frontier_rets)
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


# ==========================================
# Historical Crisis Periods for Stress Testing
# ==========================================

CRISIS_PERIODS = {
    "Dot-Com Crash (2000-2002)": {"start": "2000-03-10", "end": "2002-10-09"},
    "Global Financial Crisis (2007-2009)": {"start": "2007-10-09", "end": "2009-03-09"},
    "COVID-19 Crash (2020)": {"start": "2020-02-19", "end": "2020-03-23"},
    "2022 Bear Market": {"start": "2022-01-03", "end": "2022-10-12"},
}

# Black-Litterman default uncertainty scalar
BL_TAU_DEFAULT = 0.05

# ==========================================
# Feature: Stress Testing
# ==========================================

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def run_stress_test(tickers, weights, crises=None):
    """
    Simulates how a portfolio would have performed during historical market crises.
    
    Args:
        tickers (list): List of ticker symbols.
        weights (np.ndarray): Portfolio weights (must sum to 1).
        crises (dict): Optional custom crisis periods. Defaults to CRISIS_PERIODS.
        
    Returns:
        list[dict]: Results per crisis with drawdown, total return, and recovery info.
    """
    if crises is None:
        crises = CRISIS_PERIODS
    
    results = []
    
    for crisis_name, period in crises.items():
        try:
            # Fetch data for the crisis period (extend end by 6 months for recovery analysis)
            crisis_end = pd.Timestamp(period['end'])
            extended_end = (crisis_end + pd.DateOffset(months=6)).strftime('%Y-%m-%d')
            
            raw = yf.download(tickers, start=period['start'], end=extended_end, 
                             ignore_tz=True, progress=False)
            
            if raw is None or raw.empty:
                results.append({
                    'crisis': crisis_name,
                    'error': 'No data available for this period'
                })
                continue
            
            # Extract prices
            price_data = extract_price_data(raw, prefer_adj_close=True)
            if price_data is None or price_data.empty:
                results.append({
                    'crisis': crisis_name,
                    'error': 'Could not extract price data'
                })
                continue
            
            # Ensure we have data for all tickers
            available = [t for t in tickers if t in price_data.columns]
            if len(available) < len(tickers):
                # Reweight for available tickers
                avail_idx = [i for i, t in enumerate(tickers) if t in available]
                avail_weights = np.array([weights[i] for i in avail_idx])
                if avail_weights.sum() > 0:
                    avail_weights = avail_weights / avail_weights.sum()
                else:
                    continue
            else:
                available = tickers
                avail_weights = np.array(weights)
            
            prices = price_data[available].dropna()
            if len(prices) < 2:
                continue
            
            # Calculate portfolio daily returns
            daily_returns = prices.pct_change().dropna()
            portfolio_returns = daily_returns.values @ avail_weights
            
            # Cumulative return series
            cum_returns = (1 + portfolio_returns).cumprod()
            
            # Split into crisis period and recovery period
            crisis_end_ts = pd.Timestamp(period['end'])
            crisis_mask = daily_returns.index <= crisis_end_ts
            
            if crisis_mask.any():
                crisis_cum = cum_returns[:crisis_mask.sum()]
                total_return_crisis = float(crisis_cum[-1] / crisis_cum[0] - 1) if len(crisis_cum) > 0 else 0.0
            else:
                total_return_crisis = 0.0
            
            # Max drawdown during crisis
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            max_drawdown = float(np.min(drawdowns))
            
            # Total return over full period
            total_return_full = float(cum_returns[-1] / cum_returns[0] - 1) if len(cum_returns) > 0 else 0.0
            
            # Recovery: did portfolio recover to pre-crisis level?
            recovered = bool(cum_returns[-1] >= cum_returns[0]) if len(cum_returns) > 0 else False
            
            # Days to trough
            trough_idx = int(np.argmin(cum_returns))
            
            results.append({
                'crisis': crisis_name,
                'period': f"{period['start']} to {period['end']}",
                'total_return': total_return_crisis,
                'max_drawdown': max_drawdown,
                'trough_day': trough_idx,
                'recovered': recovered,
                'total_return_extended': total_return_full,
                'cum_returns': cum_returns,
                'dates': daily_returns.index.tolist()
            })
            
        except Exception as e:
            results.append({
                'crisis': crisis_name,
                'error': str(e)
            })
    
    return results


# ==========================================
# Feature: Black-Litterman Optimization
# ==========================================

def optimize_portfolio_black_litterman(returns, views_dict, risk_free_rate, 
                                        tau=BL_TAU_DEFAULT, num_portfolios=5000):
    """
    Black-Litterman portfolio optimization.
    
    Blends market equilibrium returns with user-specified views to produce
    adjusted expected returns, then maximizes a mean-variance (quadratic) utility.
    
    Args:
        returns (pd.DataFrame): Daily returns of assets.
        views_dict (dict): {ticker: expected_annual_return} e.g. {"AAPL": 0.15}
        risk_free_rate (float): Risk-free rate (decimal).
        tau (float): Uncertainty scaling parameter (default 0.05).
        num_portfolios (int): Number of random portfolios for visualization.
        
    Returns:
        dict: Optimization results with adjusted weights and metrics.
    """
    if returns is None or returns.empty:
        return None
    
    clean_returns = returns.dropna()
    tickers = list(clean_returns.columns)
    n_assets = len(tickers)
    
    # 1. Estimate covariance matrix (Ledoit-Wolf for robustness)
    lw = LedoitWolf()
    lw.fit(clean_returns)
    Sigma = lw.covariance_ * 252  # Annualized
    
    # 2. Equilibrium returns (reverse optimization with equal weights as proxy)
    # In practice, market-cap weights would be used; we approximate with equal weights
    delta = 2.5  # Risk aversion coefficient (standard)
    eq_weights = np.ones(n_assets) / n_assets
    pi = delta * Sigma @ eq_weights  # Equilibrium excess returns
    
    # 3. Construct views matrices (P, Q, Omega)
    view_tickers = [t for t in views_dict if t in tickers]
    if not view_tickers:
        # No valid views — fall back to equilibrium
        adjusted_returns = pi
    else:
        k = len(view_tickers)  # Number of views
        P = np.zeros((k, n_assets))  # Pick matrix
        Q = np.zeros(k)  # View returns
        
        for i, ticker in enumerate(view_tickers):
            idx = tickers.index(ticker)
            P[i, idx] = 1.0
            Q[i] = views_dict[ticker]  # Already annual
        
        # Omega: uncertainty of views (proportional to asset variance)
        Omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T))
        
        # 4. Black-Litterman formula
        tau_Sigma_inv = np.linalg.inv(tau * Sigma)
        Omega_inv = np.linalg.inv(Omega)
        
        M = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)
        adjusted_returns = M @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)
    
    # 5. Optimize: maximize mean-variance (quadratic) utility using adjusted returns
    w = cp.Variable(n_assets)
    ret = adjusted_returns @ w
    risk = cp.quad_form(w, Sigma)

    # Long-only, fully-invested constraints
    constraints = [cp.sum(w) == 1, w >= 0]

    # Quadratic utility: maximize expected return penalized by risk aversion delta.
    # This is the standard Black-Litterman objective, not a max-Sharpe objective.
    objective = cp.Maximize(ret - delta * risk)
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve()
    except cp.SolverError:
        return None
    
    if w.value is None:
        return None
    
    opt_weights = w.value.copy()
    opt_weights[opt_weights < 1e-5] = 0
    opt_weights /= opt_weights.sum()
    
    opt_ret = float(np.dot(opt_weights, adjusted_returns))
    opt_vol = float(np.sqrt(np.dot(opt_weights.T, np.dot(Sigma, opt_weights))))
    
    # 6. Random portfolios for visualization
    weights_sim = np.random.random((num_portfolios, n_assets))
    weights_sim /= np.sum(weights_sim, axis=1)[:, np.newaxis]
    
    sim_returns = np.dot(weights_sim, adjusted_returns)
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
        'tickers': tickers,
        'adjusted_returns': adjusted_returns,
        'equilibrium_returns': pi
    }


# ==========================================
# Feature: Risk Parity Optimization
# ==========================================

def optimize_portfolio_risk_parity(returns, num_portfolios=5000):
    """
    Risk Parity optimization: equal risk contribution from each asset.
    
    Each asset contributes the same amount to total portfolio variance.
    
    Args:
        returns (pd.DataFrame): Daily returns of assets.
        num_portfolios (int): Number of random portfolios for visualization.
        
    Returns:
        dict: Optimization results.
    """
    if returns is None or returns.empty:
        return None
    
    clean_returns = returns.dropna()
    n_assets = len(clean_returns.columns)
    mu = clean_returns.mean().values * 252
    
    lw = LedoitWolf()
    lw.fit(clean_returns)
    Sigma = lw.covariance_ * 252
    
    # Risk Parity: minimize sum of (RC_i - 1/N)^2
    # where RC_i = w_i * (Sigma @ w)_i / (w' Sigma w) is the risk contribution
    def risk_parity_objective(w):
        w = np.array(w)
        port_var = w.T @ Sigma @ w
        if port_var < 1e-12:
            return 1e10
        marginal_risk = Sigma @ w
        risk_contrib = w * marginal_risk / port_var
        target_rc = 1.0 / n_assets
        return np.sum((risk_contrib - target_rc) ** 2)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.001, 1.0) for _ in range(n_assets))
    init = np.ones(n_assets) / n_assets
    
    result = sco.minimize(risk_parity_objective, init,
                          method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-12})
    
    opt_weights = result.x
    opt_weights[opt_weights < 1e-5] = 0
    opt_weights /= opt_weights.sum()
    
    opt_ret = float(np.dot(opt_weights, mu))
    opt_vol = float(np.sqrt(np.dot(opt_weights.T, np.dot(Sigma, opt_weights))))
    
    # Risk contributions for display
    marginal = Sigma @ opt_weights
    port_var = opt_weights.T @ Sigma @ opt_weights
    risk_contributions = opt_weights * marginal / port_var if port_var > 0 else np.zeros(n_assets)
    
    # Random portfolios for visualization
    weights_sim = np.random.random((num_portfolios, n_assets))
    weights_sim /= np.sum(weights_sim, axis=1)[:, np.newaxis]
    
    sim_returns = np.dot(weights_sim, mu)
    sim_variances = np.einsum('ij,jk,ik->i', weights_sim, Sigma, weights_sim)
    sim_stds = np.sqrt(sim_variances)
    
    sim_stds_safe = np.where(sim_stds > MIN_VOLATILITY_FOR_SHARPE, sim_stds, MIN_VOLATILITY_FOR_SHARPE)
    sim_sharpes = (sim_returns - 0.0) / sim_stds_safe  # rf=0 for visualization
    
    results_sim = np.vstack([sim_returns, sim_stds, sim_sharpes])
    
    return {
        'results': results_sim,
        'opt_weights': opt_weights,
        'opt_ret': opt_ret,
        'opt_std': opt_vol,
        'returns': returns,
        'tickers': clean_returns.columns,
        'risk_contributions': risk_contributions
    }


# ==========================================
# Feature: Minimum-CVaR Optimization
# ==========================================

def optimize_portfolio_min_cvar(returns, risk_free_rate, num_portfolios=5000, alpha=0.95):
    """
    Minimum Conditional Value-at-Risk (CVaR) portfolio optimization.

    Minimizes the tail risk (Expected Shortfall) of the portfolio using the
    Rockafellar-Uryasev linear formulation, solved as a convex program with CVXPY.
    Historical daily returns are used directly as loss scenarios, so no Gaussian
    assumption is imposed on the tail.

    Rockafellar-Uryasev formulation:
        minimize    zeta + 1 / (S * (1 - alpha)) * sum(u_s)
        subject to  u_s >= loss_s - zeta,  u_s >= 0
                    sum(w) = 1,  w >= 0
    where loss_s = -(R_s . w) is the portfolio loss in scenario s and zeta is the
    Value-at-Risk at confidence level alpha.

    Args:
        returns (pd.DataFrame): Daily returns of the assets.
        risk_free_rate (float): The risk-free rate (decimal), used for the scatter plot.
        num_portfolios (int): Number of random portfolios to generate for visualization.
        alpha (float): CVaR confidence level (default 0.95 -> 5% worst tail).

    Returns:
        dict: Optimization results containing weights, metrics, and simulation data,
              plus 'cvar' (expected daily shortfall of the optimal portfolio) and 'alpha'.
    """
    if returns is None or returns.empty:
        return None

    clean_returns = returns.dropna()
    if clean_returns.shape[0] < 2 or clean_returns.shape[1] < 2:
        return None

    R = clean_returns.values  # (S scenarios, n assets) of daily returns
    n_scenarios, n_assets = R.shape

    # Annualized mean returns and Ledoit-Wolf covariance (for plotting consistency)
    mu = clean_returns.mean().values * 252
    lw = LedoitWolf()
    lw.fit(clean_returns)
    Sigma = lw.covariance_ * 252

    # Convex program (Rockafellar-Uryasev)
    w = cp.Variable(n_assets)
    zeta = cp.Variable()
    u = cp.Variable(n_scenarios, nonneg=True)

    portfolio_losses = -R @ w
    constraints = [
        u >= portfolio_losses - zeta,
        cp.sum(w) == 1,
        w >= 0,
    ]
    cvar_expr = zeta + (1.0 / (n_scenarios * (1.0 - alpha))) * cp.sum(u)
    prob = cp.Problem(cp.Minimize(cvar_expr), constraints)

    try:
        prob.solve()
    except cp.SolverError:
        return None

    if w.value is None:
        return None

    # Clean and renormalize weights
    opt_weights = np.asarray(w.value).flatten()
    opt_weights[opt_weights < 1e-5] = 0
    if opt_weights.sum() <= 0:
        return None
    opt_weights /= opt_weights.sum()

    opt_ret = float(np.dot(opt_weights, mu))
    opt_vol = float(np.sqrt(np.dot(opt_weights.T, np.dot(Sigma, opt_weights))))

    # Realized daily CVaR (expected shortfall) of the optimal portfolio
    port_daily = R @ opt_weights
    var_threshold = np.percentile(port_daily, (1.0 - alpha) * 100)
    tail = port_daily[port_daily <= var_threshold]
    cvar_daily = float(-np.mean(tail)) if tail.size > 0 else float(-var_threshold)

    # Random portfolios for visualization (Ledoit-Wolf Sigma for consistency)
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
        'tickers': clean_returns.columns,
        'cvar': cvar_daily,
        'alpha': alpha,
    }


# ==========================================
# Feature: Downside Risk Metrics
# ==========================================

def downside_deviation(returns, target=0.0, periods_per_year=252):
    """
    Annualized downside deviation: standard deviation of returns that fall below a target.

    Unlike standard deviation, only negative deviations (below the target) are penalized,
    which better reflects an investor's perception of "bad" risk.

    Args:
        returns (array-like): Periodic (e.g. daily) returns.
        target (float): Minimum acceptable periodic return (default 0.0).
        periods_per_year (int): Annualization factor (default 252 trading days).

    Returns:
        float: Annualized downside deviation.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return 0.0
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    return float(dd * np.sqrt(periods_per_year))


def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Annualized Sortino ratio: excess return per unit of downside deviation.

    A variant of the Sharpe ratio that divides by downside deviation instead of total
    volatility, rewarding strategies that limit losses rather than penalizing upside.

    Args:
        returns (array-like): Periodic (e.g. daily) returns.
        risk_free_rate (float): Annual risk-free rate (decimal).
        periods_per_year (int): Annualization factor (default 252).

    Returns:
        float: Annualized Sortino ratio (0.0 if downside deviation is ~0).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return 0.0
    daily_rf = risk_free_rate / periods_per_year
    ann_excess = float(np.mean(r - daily_rf) * periods_per_year)
    dd = downside_deviation(r, target=daily_rf, periods_per_year=periods_per_year)
    if dd < MIN_VOLATILITY_FOR_SHARPE:
        return 0.0
    return float(ann_excess / dd)


def calmar_ratio(cagr, max_drawdown):
    """
    Calmar ratio: compound annual growth rate divided by the absolute maximum drawdown.

    Measures return earned per unit of worst-case peak-to-trough loss.

    Args:
        cagr (float): Compound annual growth rate (decimal).
        max_drawdown (float): Maximum drawdown (negative decimal, e.g. -0.25).

    Returns:
        float: Calmar ratio (0.0 if max drawdown is ~0).
    """
    if max_drawdown is None or abs(max_drawdown) < MIN_VOLATILITY_FOR_SHARPE:
        return 0.0
    return float(cagr / abs(max_drawdown))


def omega_ratio(returns, threshold=0.0):
    """
    Omega ratio: probability-weighted ratio of gains to losses relative to a threshold.

    Captures the full shape of the return distribution (all moments), not just mean and
    variance. Omega > 1 means gains above the threshold outweigh losses below it.

    Args:
        returns (array-like): Periodic returns.
        threshold (float): Periodic threshold return (default 0.0).

    Returns:
        float: Omega ratio (inf if there are gains but no losses).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return 0.0
    gains = float(np.sum(np.maximum(r - threshold, 0.0)))
    losses = float(np.sum(np.maximum(threshold - r, 0.0)))
    if losses < MIN_VOLATILITY_FOR_SHARPE:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)


# ==========================================
# Feature: Portfolio Backtesting
# ==========================================

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def run_backtest(tickers, weights, start_date, end_date, rebal_freq='none', benchmark='SPY'):
    """
    Backtests a portfolio over a historical date range.
    
    Args:
        tickers (list): List of ticker symbols.
        weights (np.ndarray): Target portfolio weights.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        rebal_freq (str): Rebalancing frequency: 'none', 'monthly', 'quarterly', 'annually'.
        benchmark (str): Benchmark ticker for comparison.
        
    Returns:
        dict: Backtest results with cumulative returns, metrics, and comparison data.
    """
    # Fetch portfolio data
    all_tickers = list(tickers) + ([benchmark] if benchmark and benchmark not in tickers else [])
    raw = yf.download(all_tickers, start=start_date, end=end_date, ignore_tz=True, progress=False)
    
    if raw is None or raw.empty:
        return None
    
    price_data = extract_price_data(raw, prefer_adj_close=True)
    if price_data is None or price_data.empty:
        return None
    
    # Separate benchmark
    bench_returns = None
    if benchmark and benchmark in price_data.columns:
        bench_prices = price_data[benchmark]
        bench_returns = bench_prices.pct_change().dropna()
        bench_cum = (1 + bench_returns).cumprod()
    
    # Portfolio prices & returns
    available = [t for t in tickers if t in price_data.columns]
    if len(available) == 0:
        return None
    
    avail_idx = [i for i, t in enumerate(tickers) if t in available]
    port_weights = np.array([weights[i] for i in avail_idx])
    port_weights = port_weights / port_weights.sum()
    
    port_prices = price_data[available].dropna()
    daily_returns = port_prices.pct_change().dropna()
    
    # Compute weighted returns (with optional periodic rebalancing)
    if rebal_freq == 'none':
        weighted_returns = daily_returns.values @ port_weights
    else:
        # For rebalanced: recalculate by resetting weights at each rebal point
        freq_map = {'monthly': 'ME', 'quarterly': 'QE', 'annually': 'YE'}
        rebal_dates_set = set(pd.date_range(start=daily_returns.index[0],
                                             end=daily_returns.index[-1],
                                             freq=freq_map.get(rebal_freq, 'ME')).date)
        
        current_w = port_weights.copy()
        weighted_returns = np.zeros(len(daily_returns))
        
        for i in range(len(daily_returns)):
            day_ret = daily_returns.iloc[i].values
            weighted_returns[i] = np.dot(current_w, day_ret)
            
            # Update weights based on return drift
            current_w = current_w * (1 + day_ret)
            w_sum = current_w.sum()
            if w_sum > 0:
                current_w = current_w / w_sum
            else:
                current_w = port_weights.copy()  # Reset if degenerate
            
            # Rebalance
            if daily_returns.index[i].date() in rebal_dates_set:
                current_w = port_weights.copy()
    
    port_cum = (1 + pd.Series(weighted_returns, index=daily_returns.index)).cumprod()
    
    # Compute metrics
    total_return = float(port_cum.iloc[-1] - 1)
    n_years = len(daily_returns) / 252.0
    
    # Guard CAGR: handle edge case where portfolio value goes to zero or negative
    if port_cum.iloc[-1] > 0 and n_years > 0:
        cagr = float((port_cum.iloc[-1]) ** (1 / n_years) - 1)
    else:
        cagr = -1.0  # Total loss
    
    ann_vol = float(np.std(weighted_returns) * np.sqrt(252))
    sharpe = cagr / ann_vol if ann_vol > MIN_VOLATILITY_FOR_SHARPE else 0.0
    
    # Max drawdown
    running_max = port_cum.cummax()
    drawdown_series = (port_cum - running_max) / running_max
    max_dd = float(drawdown_series.min())
    
    # Win rate
    win_rate = float(np.mean(weighted_returns > 0))

    # Downside risk metrics
    downside_dev = downside_deviation(weighted_returns)
    sortino = sortino_ratio(weighted_returns)
    calmar = calmar_ratio(cagr, max_dd)

    # Build result
    result = {
        'portfolio_cum': port_cum,
        'dates': daily_returns.index,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'downside_deviation': downside_dev,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'drawdown_series': drawdown_series
    }
    
    if bench_returns is not None:
        # Align dates
        common_dates = port_cum.index.intersection(bench_cum.index)
        if len(common_dates) > 0:
            result['benchmark_cum'] = bench_cum.loc[common_dates]
            bench_total = float(bench_cum.loc[common_dates].iloc[-1] - 1)
            result['benchmark_return'] = bench_total
            result['benchmark_name'] = benchmark
    
    return result


# ==========================================
# Feature: Rolling Risk Metrics
# ==========================================

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def compute_rolling_metrics(price_data, window=60, benchmark_ticker='SPY'):
    """
    Computes rolling risk metrics for a portfolio or single asset.
    
    Args:
        price_data (pd.Series or pd.DataFrame): Price series.
        window (int): Rolling window in trading days.
        benchmark_ticker (str): Benchmark ticker for rolling beta.
        
    Returns:
        dict: Rolling volatility, Sharpe ratio, and beta series.
    """
    if price_data is None or (hasattr(price_data, 'empty') and price_data.empty):
        return None
    
    # If DataFrame, compute equally-weighted portfolio returns
    if isinstance(price_data, pd.DataFrame):
        returns = price_data.pct_change().dropna()
        port_returns = returns.mean(axis=1)  # Equal weight
    else:
        port_returns = price_data.pct_change().dropna()
    
    # Rolling volatility (annualized)
    rolling_vol = port_returns.rolling(window=window).std() * np.sqrt(252)
    
    # Rolling Sharpe (annualized, rf=0)
    rolling_mean = port_returns.rolling(window=window).mean() * 252
    rolling_sharpe = rolling_mean / rolling_vol
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)

    # Rolling Sortino (annualized, rf=0): rolling mean / rolling downside deviation
    downside = port_returns.where(port_returns < 0, 0.0)
    rolling_downside_dev = np.sqrt((downside ** 2).rolling(window=window).mean()) * np.sqrt(252)
    rolling_sortino = (rolling_mean / rolling_downside_dev).replace([np.inf, -np.inf], np.nan)

    result = {
        'rolling_vol': rolling_vol.dropna(),
        'rolling_sharpe': rolling_sharpe.dropna(),
        'rolling_sortino': rolling_sortino.dropna(),
        'dates': rolling_vol.dropna().index
    }
    
    # Rolling beta (vs benchmark)
    try:
        bench_data = yf.download(benchmark_ticker, start=port_returns.index[0], 
                                  end=port_returns.index[-1], ignore_tz=True, progress=False)
        if bench_data is not None and not bench_data.empty:
            bench_price = extract_price_data(bench_data, prefer_adj_close=True)
            if bench_price is not None:
                bench_returns = bench_price.iloc[:, 0].pct_change().dropna()
                
                # Align dates
                common = port_returns.index.intersection(bench_returns.index)
                pr = port_returns.loc[common]
                br = bench_returns.loc[common]
                
                # Rolling beta = Cov(port, bench) / Var(bench)
                rolling_cov = pr.rolling(window=window).cov(br)
                rolling_var = br.rolling(window=window).var()
                rolling_beta = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan).dropna()
                
                result['rolling_beta'] = rolling_beta
                result['benchmark_name'] = benchmark_ticker
    except Exception:
        pass  # Beta is optional
    
    return result


# ==========================================
# Feature: Factor Decomposition (CAPM)
# ==========================================

def compute_factor_decomposition(portfolio_returns, benchmark_returns, risk_free_rate=0.0):
    """
    Perform CAPM factor decomposition: regress portfolio returns against benchmark.
    
    Args:
        portfolio_returns (pd.Series): Daily portfolio returns.
        benchmark_returns (pd.Series): Daily benchmark returns.
        risk_free_rate (float): Annual risk-free rate (decimal).
        
    Returns:
        dict: Alpha (annualized), Beta, R², Tracking Error, Information Ratio.
    """
    # Align dates
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 30:
        return None
    
    pr = portfolio_returns.loc[common].values
    br = benchmark_returns.loc[common].values
    
    daily_rf = risk_free_rate / 252.0
    
    # Excess returns
    pr_excess = pr - daily_rf
    br_excess = br - daily_rf
    
    # Linear regression: R_p - R_f = alpha + beta * (R_m - R_f) + epsilon
    model = LinearRegression()
    model.fit(br_excess.reshape(-1, 1), pr_excess)
    
    beta = float(model.coef_[0])
    daily_alpha = float(model.intercept_)
    alpha_annualized = daily_alpha * 252  # Annualize
    r_squared = float(model.score(br_excess.reshape(-1, 1), pr_excess))
    
    # Tracking error: std of (portfolio - benchmark) returns, annualized
    active_returns = pr - br
    tracking_error = float(np.std(active_returns) * np.sqrt(252))
    
    # Information ratio: annualized active return / tracking error
    active_return_ann = float(np.mean(active_returns) * 252)
    information_ratio = active_return_ann / tracking_error if tracking_error > MIN_VOLATILITY_FOR_SHARPE else 0.0
    
    return {
        'alpha': alpha_annualized,
        'beta': beta,
        'r_squared': r_squared,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio
    }
