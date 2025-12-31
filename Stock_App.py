import numpy as np
import pandas as pd
import plotly.graph_objects as go 
import scipy.optimize as sco 
import streamlit as st
import yfinance as yf

# ==========================================
# Application Configuration
# ==========================================

st.set_page_config(page_title = "Financial Analysis Tool", 
                   layout = "wide",
                   initial_sidebar_state = "collapsed")

# Inject custom CSS for metric styling
st.markdown("""
            <style>
                div[data-testid = "stMetricValue"] {font-size: 24px;}
            </style>
            """, 
            unsafe_allow_html = True)

# ==========================================
# Main Page Header
# ==========================================

st.title("Financial Analysis Tool")
st.caption("Advanced Financial Modeling & Optimization")

# Page Navigation
page = st.radio("Select Tool:", 
                ["Stock Price Forecaster", "Portfolio Optimizer", "Portfolio Rebalancer"], 
                horizontal = True,
                label_visibility = "collapsed")

st.markdown("---")

# ==========================================
# Load Data Using Yahoo Finance API
# ==========================================

@st.cache_data(ttl = 3600)
def get_stock_data(tickers, period):
    """
    Fetches historical stock data from Yahoo Finance for a single ticker or a list of tickers.
    
    Optimized to use yf.download() consistently for both single and multiple tickers,
    which is more efficient and returns consistent data structures.

    Args:
        tickers (str or list[str]): A single ticker symbol (e.g., "AAPL") or a list of symbols (e.g., ["AAPL", "GOOG"]).
        period (str): The historical period to download (e.g., "1y", "3y", "max").

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing historical stock data (Open, High, Low, Close, Volume) if successful; 
                              None if an API error occurs.
    """
    try:
        # Standardize to list for consistent processing
        ticker_list = [tickers] if isinstance(tickers, str) else tickers
        
        # Use yf.download() for both single and multiple tickers (more efficient and consistent)
        # This returns MultiIndex columns for both single and multiple tickers
        data = yf.download(ticker_list, period=period, ignore_tz=True, progress=False, show_errors=False)
        
        # Handle empty or invalid data
        if data is None or data.empty:
            return None
        
        # Return raw data - let extract_price_data() handle column extraction consistently
        return data
            
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None


@st.cache_data(ttl = 3600)
def get_stock_info(ticker):
    """
    Fetches stock information (company name, etc.) from Yahoo Finance.
    
    Cached to avoid redundant API calls when the same ticker is requested multiple times.

    Args:
        ticker (str): A single ticker symbol (e.g., "AAPL").

    Returns:
        dict or None: Stock information dictionary if successful; None if an API error occurs.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info if info else None
    except Exception as e:
        # Silently fail - info is optional, not critical for functionality
        return None


def extract_price_data(raw_data, prefer_adj_close=True):
    """
    Extracts price data from raw Yahoo Finance data, handling both single and multi-ticker formats.
    
    Args:
        raw_data (pd.DataFrame): Raw data from get_stock_data()
        prefer_adj_close (bool): If True, prefer 'Adj Close' over 'Close'
    
    Returns:
        pd.DataFrame: DataFrame with price data (one column per ticker)
    """
    if raw_data is None or raw_data.empty:
        return None
    
    # Handle MultiIndex columns (from batch downloads)
    if isinstance(raw_data.columns, pd.MultiIndex):
        # Extract price column (prefer Adj Close)
        price_col = 'Adj Close' if prefer_adj_close else 'Close'
        if price_col in raw_data.columns.get_level_values(0):
            data = raw_data[price_col]
            # Flatten column names to just ticker symbols
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(1)
        else:
            # Fallback to 'Close' if 'Adj Close' not available
            price_col = 'Close'
            if price_col in raw_data.columns.get_level_values(0):
                data = raw_data[price_col]
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(1)
            else:
                return None
    else:
        # Single ticker or already flattened
        price_col = 'Adj Close' if (prefer_adj_close and 'Adj Close' in raw_data.columns) else 'Close'
        if price_col in raw_data.columns:
            data = raw_data[[price_col]]  # Keep as DataFrame
        else:
            return None
    
    # Drop columns that are entirely NaN (invalid tickers)
    data = data.dropna(axis=1, how='all')
    
    return data if not data.empty else None

# ==========================================
# MODULE 1: STOCK PRICE FORECASTER
# ==========================================

if page == "Stock Price Forecaster":
    
    st.header("Stock Price Prediction")
    
    # --- Sidebar Settings ---  
    st.sidebar.header("Simulation Parameters")
    
    time_horizon = st.sidebar.slider("Time Horizon (Trading Days)", 
                                     5, 365, 30,
                                     help = "Number of trading days into the future for prediction")
    
    simulations = st.sidebar.slider("Number of Simulations", 
                                    100, 1000, 200,
                                    help = "More simulations = more accurate results, but slower speed")
    
    seed = st.sidebar.number_input("Random Seed", 
                                   value = 42, 
                                   min_value = 0,
                                   step = 1,
                                   format = "%d",
                                   help = "Fix the random numbers for reproducible results.")
    
    # --- Input Section ---
    col1, col2 = st.columns([4, 1]) 
    
    with col1:
        ticker = st.text_input("Enter Stock Ticker", 
                               value = "VOO", 
                               help = "Please enter a valid ticker (e.g., VOO, AAPL)")
    with col2:
        st.write("") 
        st.write("") 
        start_sim = st.button("Start Simulation", 
                              type = "primary", 
                              use_container_width = True)
    
    # --- Simulation Logic ---
    if start_sim:
        with st.spinner('Running Monte Carlo Simulation...'):

            np.random.seed(int(round(seed)))
            stock_data = get_stock_data(ticker, period = "1y")

            if stock_data is None or stock_data.empty:
                st.error(f"Ticker '{ticker}' not found or API unavailable.")
                
            else:

                # Fetch full name using cached function
                stock_info = get_stock_info(ticker)
                stock_name = stock_info.get('longName', ticker) if stock_info else ticker

                # Data Preprocessing using helper function
                price_data = extract_price_data(stock_data, prefer_adj_close=True)
                
                if price_data is None or price_data.empty:
                    st.error(f"Data Error: Closing price is missing for {ticker}.")
                    st.stop()
                
                # Extract as Series for single ticker
                closing_prices = price_data.iloc[:, 0]  # Get first (and only) column as Series

                last_price = float(closing_prices.iloc[-1]) 

                # Calculate Daily Change (Previous Close vs Current)
                if len(closing_prices) >= 2:
                    prev_price = float(closing_prices.iloc[-2])
                    price_change = last_price - prev_price
                    pct_change = (price_change / prev_price) * 100
                else:
                    price_change = 0.0
                    pct_change = 0.0

                # Calculate Log Returns for Geometric Brownian Motion parameters
                log_returns = np.log(closing_prices / closing_prices.shift(1)).dropna()
                mu = log_returns.mean()
                sigma = log_returns.std()
                
                # --- Vectorized Monte Carlo Simulation (GBM) ---
                # 1. Pre-compute random shocks (Brownian Motion component)
                # Shape: (time_horizon, simulations)
                shocks = np.random.normal(0, 1, (time_horizon, simulations))
            
                # 2. Compute Daily Returns using Geometric Brownian Motion formula
                drift = mu - 0.5 * sigma**2
                daily_returns = np.exp(drift + sigma * shocks)
            
                # 3. Aggregate into Price Paths
                # Initialize starting point with 1s to apply last_price scaling later
                price_paths = np.vstack([np.ones((1, simulations)), daily_returns])
                price_paths = last_price * price_paths.cumprod(axis = 0)

                # Compute Key Metrics directly from numpy array (memory efficient)
                end_prices = price_paths[-1, :]  # Last row contains all terminal prices

                # Center tendency
                expected_price = float(np.mean(end_prices))
                median_price = float(np.median(end_prices))
                
                # For worst case, we use a 95% CI              
                worst_case = float(np.percentile(end_prices, 5))

                # CVaR / Expected Shortfall (average of worst 5%)
                tail = end_prices[end_prices <= worst_case]
                cvar_95 = float(np.mean(tail)) if len(tail) > 0 else worst_case  # fallback safeguard
                
                # Probability of Loss
                prob_loss = float(np.mean(end_prices < last_price))  # 0~1
                
                # Optimization: Only create DataFrame for visualization subset (max 50 columns)
                # This reduces memory usage by ~95% when running 1000 simulations
                max_lines_to_plot = 50
                columns_to_store = min(simulations, max_lines_to_plot)
                
                # Find the worst scenario index to ensure it's included in the plot
                worst_scenario_idx = int(np.argmin(np.abs(end_prices - worst_case)))
                
                # Select columns to store: first N columns + worst scenario if not already included
                columns_indices = list(range(columns_to_store))
                if worst_scenario_idx not in columns_indices and worst_scenario_idx < simulations:
                    # Replace last column with worst scenario if it's not in the first N
                    columns_indices[-1] = worst_scenario_idx
                
                # Compute mean path from full array for accurate visualization
                mean_path_full = np.mean(price_paths, axis=1)
                
                # Create DataFrame with only the subset needed for visualization + mean path
                subset_data = np.column_stack([price_paths[:, columns_indices], mean_path_full])
                subset_columns = [f"Sim_{i}" for i in columns_indices] + ['Mean']
                simulation_df = pd.DataFrame(subset_data, 
                                            columns = subset_columns,
                                            index = range(len(price_paths)))

                # --- SAVE TO SESSION STATE ---
                st.session_state['forecast_results'] = {'simulation_df': simulation_df,
                                                        'last_price': last_price,
                                                        'stock_name': stock_name,      
                                                        'price_change': price_change,  
                                                        'pct_change': pct_change,    
                                                        'expected_price': expected_price,
                                                        'median_price': median_price,
                                                        'worst_case': worst_case,
                                                        'cvar_95': cvar_95,
                                                        'prob_loss': prob_loss,
                                                        'ticker': ticker,
                                                        'time_horizon': time_horizon, 
                                                        'simulations': simulations}

    if 'forecast_results' in st.session_state:
        
        # Retrieve data
        res = st.session_state['forecast_results']
        simulation_df = res['simulation_df']
        last_price = res['last_price']
        saved_name = res['stock_name']
        saved_change = res['price_change']
        saved_pct = res['pct_change']
        expected_price = res['expected_price']
        median_price = res['median_price']
        worst_case = res['worst_case']
        cvar_95 = res['cvar_95']
        prob_loss = res['prob_loss']
        saved_ticker = res['ticker'] 
        saved_horizon = res['time_horizon']
        saved_sims = res['simulations']
    
        # --- Output Visualization ---
        st.write("") # Spacing
        col_header1, col_header2 = st.columns([3, 1]) # Left gets more space

        with col_header1:
            st.markdown(f"<h1 style='margin-bottom:0px;'>{saved_ticker}</h1>", unsafe_allow_html = True)
            st.caption(f"{saved_name}") # Small font for full name
        
        with col_header2:
            st.write("") 
            st.metric(label = "Current Price", 
                      value = f"${last_price:.2f}", 
                      delta = f"{saved_change:+.2f} ({saved_pct:+.2f}%)")
            
        st.markdown("---")

        # Initiate the figure
        fig = go.Figure()
                
        # All columns in simulation_df are already optimized subset (max 50)
        # The worst scenario is already included during DataFrame creation
        columns_to_plot = list(simulation_df.columns)

        # Drawing the plot (exclude 'Mean' column from individual traces)
        sim_columns = [col for col in columns_to_plot if col != 'Mean']
        for col in sim_columns:
            fig.add_trace(go.Scatter(x = simulation_df.index,
                                     y = simulation_df[col],
                                     mode = 'lines', 
                                     opacity = 0.375,
                                     line = dict(width = 1),
                                     showlegend = False,
                                     hoverinfo = 'skip' ))
    
        # Add Mean Expectation Line (precomputed from full array)
        if 'Mean' in simulation_df.columns:
            fig.add_trace(go.Scatter(x = simulation_df.index,
                                     y = simulation_df['Mean'],
                                     mode = 'lines',
                                     name = 'Expected Average',
                                     line = dict(color = 'red', width = 3),
                                     opacity = 1.0))
                
        # Layout setting
        fig.update_layout(title = f"{saved_sims} Monte Carlo Simulations Scenarios",
                          xaxis_title = "Trading Days into Future",
                          yaxis_title = "Price (USD)",
                          xaxis = dict(range = [0, saved_horizon]),
                          hovermode = "x unified")

        # Render
        st.plotly_chart(fig, use_container_width = True)
                
        # Guide: Interpretation
        with st.expander("ℹ️ How to interpret this chart?"):
            st.write(
                """
                This chart shows possible future price paths based on historical volatility.
                - **Red Line**: The average expected price trend.
                - **Faint Lines**: Individual simulated trajectories representing possible market scenarios.
                - **Dispersion**: A wider spread of lines indicates higher historical volatility and greater uncertainty.
                """)

        # Guide: Methodology
        with st.expander("🧠 How does the prediction work? (Methodology)"):
            st.markdown(
                r"""
                ### The Model: Geometric Brownian Motion (GBM)
                We use a statistical method standard in quantitative finance called **Geometric Brownian Motion**.
                        
                **How it works:**
                1.  **Drift ($\mu$):** We calculate the average daily return of the stock over the past year. This sets the general "trend".
                2.  **Volatility ($\sigma$):** We measure how much the stock price typically swings (standard deviation).
                3.  **Random Shock:** For every future day, we add a random value (Gaussian noise) to simulate unpredictable market news.
                        
                **The Formula:**
                The price at time $t$ is calculated as: $S_t = S_{t-1} \cdot e^{(\mu - \frac{1}{2}\sigma^2) + \sigma \cdot Z}$
                        
                *(Where $Z$ is a random number from a standard normal distribution)*
                """)

        # --- Statistical Analysis ---
        st.divider()
        st.subheader("Risk Analysis")
                
        # ROI Calculation
        expected_pct = (expected_price - last_price) / last_price * 100
        median_pct = (median_price   - last_price) / last_price * 100
        worst_pct = (worst_case - last_price) / last_price * 100
        cvar_pct = (cvar_95 - last_price) / last_price * 100

        # Setup Layout (2 Columns by 2 Rows)
        col1, col2 = st.columns(2)

        # Display Metrics
        col1.metric("Expected Price (Average)", 
                    f"${expected_price:.2f}", 
                    f"{expected_pct:+.2f}%")

        col2.metric("Median Price (50th Percentile)",
                    f"${median_price:.2f}", 
                    f"{median_pct:+.2f}%")

        col3, col4 = st.columns(2)
                
        col3.metric("Value at Risk (95% Confidence)",
                    f"${worst_case:.2f}", 
                    f"{worst_pct:+.2f}%",
                    help = "5th Percentile outcome. Indicates a 95% probability that price remains above this level.")

        col4.metric("CVaR / Expected Shortfall (95%)",
                    f"${cvar_95:.2f}",
                    f"{cvar_pct:+.2f}%",
                    help = "Average terminal price within the worst 5% outcomes. This describes tail severity beyond VaR.")
                
        st.metric("Probability of Loss",
                  f"{prob_loss*100:.1f}%",
                  help = "Share of simulations where the terminal price finishes below the current price.")

# ==========================================
# MODULE 2: PORTFOLIO OPTIMIZER (MPT)
# ==========================================

elif page == "Portfolio Optimizer":
    st.header("Efficient Frontier (Modern Portfolio Theory)")
    
    # --- Sidebar Settings ---  
    st.sidebar.header("Optimization Settings")
    
    num_portfolios = st.sidebar.slider("Number of Simulations", 
                                       1000, 10000, 5000,
                                       help = "Higher simulations produce a more accurate Efficient Frontier.")

    # Risk-Free Rate Input
    risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate (%)",
                                                   value = 3.0,
                                                   min_value = 0.0,
                                                   max_value = 10.0,
                                                   step = 0.1,
                                                   help = "Current annual risk-free rate (e.g., 3-month US Treasury Bill).")
    
    # Convert to decimal
    risk_free_rate = risk_free_rate_input / 100

    seed = st.sidebar.number_input("Random Seed", 
                                   value = 42, 
                                   min_value = 0,
                                   step = 1,
                                   format = "%d",
                                   help = "Fix the random numbers for reproducible results.")

    # --- Input Section ---  
    col_input, col_btn = st.columns([4, 1]) 
    
    with col_input:
        tickers_input = st.text_area("Enter Stock Tickers (Comma Separated)", 
                                     value = "VTI, VEA, VNQ",
                                     help = "Please enter at least 2 tickers to analyze diversification effects.")
        
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        tickers = list(set(tickers))

    with col_btn:
        st.write("") 
        st.write("") 
        start_opt = st.button("Optimize", 
                              type = "primary", 
                              use_container_width = True)

    # --- Optimization Logic ---
    if start_opt:
        
        # --- 1. Early Validation ---
        # Check before making API calls to save time
        if len(tickers) < 2:
            st.error("Error: At least 2 valid tickers are required for portfolio optimization.")
            st.stop()
            
        with st.spinner('Calculating Efficient Frontier...'):
            np.random.seed(int(round(seed)))
            # Fetch data for 3 years to calculate correlation matrix
            raw_data = get_stock_data(tickers, period = "3y")

            # Check if API returned any data
            if raw_data is None or raw_data.empty:
                st.error("Error: No data found. Please check your tickers.")
                st.stop()
                
            # --- 2. Data Cleaning & Selection ---
            try:
                # Use helper function to extract price data consistently
                data = extract_price_data(raw_data, prefer_adj_close=True)
                
                if data is None:
                    st.error("Data Error: Unable to extract price data from API response.")
                    st.stop()
                
                # Validation: Need at least 2 valid stocks for portfolio optimization
                if data.shape[1] < 2:
                    st.error("Error: Insufficient valid data. At least 2 valid stocks are needed to calculate correlation.")
                    st.stop()

            except Exception as e:
                st.error(f"An unexpected error occurred during data processing: {e}")
                st.stop()

            else:
                # --- MPT Calculations ---
                returns = data.pct_change()
                mean_returns = returns.mean() * 252 # Annualized
                cov_matrix = returns.cov() * 252    # Annualized
                    
                # Objective Functions for Scipy Optimize
                def portfolio_performance(weights, mean_returns, cov_matrix):                        
                    returns = np.sum(mean_returns * weights)
                    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    return returns, std

                def neg_sharpe(weights, mean_returns, cov_matrix, rf_rate):
                    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
                    return - (p_ret - rf_rate) / p_std

                # SLSQP Optimization for Max Sharpe Ratio
                num_assets = len(data.columns)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) 
                bounds = tuple((0, 1) for _ in range(num_assets)) 
                init_guess = num_assets * [1. / num_assets]
                
                opt_results = sco.minimize(neg_sharpe, init_guess, 
                                           args = (mean_returns, cov_matrix, risk_free_rate), 
                                           method = 'SLSQP', 
                                           bounds = bounds, 
                                           constraints = constraints)
                
                opt_weights = opt_results.x
                opt_ret, opt_std = portfolio_performance(opt_weights, mean_returns, cov_matrix)
                    
                # --- Vectorized Portfolio Simulation ---
                # 1. Generate random weight matrix (num_portfolios x num_assets)
                weights = np.random.random((num_portfolios, num_assets))
                weights /= np.sum(weights, axis = 1)[:, np.newaxis] # Normalize to sum=1

                 # 2. Compute Returns (Dot Product)
                sim_returns = np.dot(weights, mean_returns)

                # 3. Compute Volatility using Einstein Summation (einsum)
                # Efficient calculation of diagonal elements of (w @ Sigma @ w.T)
                sim_variances = np.einsum('ij,jk,ik->i', weights, cov_matrix, weights)
                sim_stds = np.sqrt(sim_variances)

                # 4. Compute Sharpe Ratios
                sim_sharpes = (sim_returns - risk_free_rate) / sim_stds

                # Stack results: [Returns, Volatility, Sharpe]
                results = np.vstack([sim_returns, sim_stds, sim_sharpes])

                # --- SAVE TO SESSION STATE ---
                st.session_state['mpt_results'] = {'results': results,
                                                   'opt_std': opt_std,
                                                   'opt_ret': opt_ret,
                                                   'opt_weights': opt_weights,
                                                   'tickers': data.columns,
                                                   'returns': returns,
                                                   'rf_rate': risk_free_rate}

    if 'mpt_results' in st.session_state:
        
        # Retrieve data
        data_store = st.session_state['mpt_results']
        results = data_store['results']
        opt_std = data_store['opt_std']
        opt_ret = data_store['opt_ret']
        opt_weights = data_store['opt_weights']
        cols = data_store['tickers']
        returns = data_store['returns']
        saved_rf = data_store['rf_rate']
                
        # --- Visualization ---
        fig = go.Figure()
                    
        # Scatter plot of random portfolios
        fig.add_trace(go.Scatter(x = results[1,:],
                                 y = results[0,:],
                                 mode = 'markers',
                                 marker = dict(color = results[2,:],
                                               colorscale = 'Viridis',
                                               showscale = True,
                                               size = 5,
                                               colorbar = dict(title = "Sharpe<br>Ratio")),
                                 name = 'Random Portfolios'))
                    
        # Highlight Optimal Portfolio
        fig.add_trace(go.Scatter(x = [opt_std],
                                 y = [opt_ret],
                                 mode = 'markers',
                                 marker = dict(color = 'red',
                                               size = 15,
                                               symbol = 'star',
                                               line = dict(color = 'white',
                                                           width = 1)),
                                 name='Max Sharpe (Optimal)'))

        fig.update_layout(title = f"Risk vs. Return Analysis (Risk-Free Rate: {saved_rf*100:.1f}%)",
                          xaxis_title = "Volatility (Annualized Std Dev)",
                          yaxis_title = "Expected Annual Return",
                          template = "plotly_dark",
                          height = 600,
                          legend = dict(yanchor = "top", y = 0.99,
                                        xanchor = "left", x = 0.01,
                                        bgcolor = "rgba(0,0,0,0.5)"))
                
        st.plotly_chart(fig, use_container_width = True)
                    
        # Guide: Interpretation
        col_info1, col_info2 = st.columns(2)
                    
        with col_info1:
            with st.expander("ℹ️ How to interpret this chart?"):
                st.markdown(
                    """
                    ### Understanding the Axes
                            
                    **1. Y-Axis: Expected Return (Profit)**
                    * The number represents the estimated annual growth rate.
                    * *Example:* 0.2 means the estimated annual growth is 20%.
                            
                    **2. X-Axis: Volatility (Risk)**
                    * **Higher X value = Wider Range = More uncertainty.**
                    * The number represents the "Swing Range" (Standard Deviation).
                    * *Example:* A return of 10% with 0.15 (15%) volatility means actual return will likely fall between **-5%** and **+25%**.
                            
                    **3. Color Scale: Sharpe Ratio**
                    * It measures return per unit of risk.
                    * **> 1.0**: Good.
                    * **> 2.0**: Very Good.
                    * **> 3.0**: Excellent.
                    """)

        with col_info2:
            with st.expander("🧠 How does the optimization work? (Methodology)"):
                st.markdown(
                    r"""
                    ### Modern Portfolio Theory (MPT)
                    We use the **Markowitz Mean-Variance Optimization** method.
                            
                    **The Logic:**
                    We simulate thousands of random combinations to find the "Efficient Frontier", the curve where you get the **maximum possible return** for a given level of risk.
                            
                    **The Goal:**
                    Maximize the **Sharpe Ratio**:
                    $$
                    \text{Sharpe} = \frac{R_p - R_f}{\sigma_p}
                    $$
                    - $R_p$: Portfolio Return
                    - $R_f$: Risk-Free Rate
                    - $\sigma_p$: Portfolio Risk (Volatility)
                    """)

        # --- Correlation Analysis & Warning System ---
                
        # 1. Calculate Correlation Matrix
        corr_matrix = returns.corr()
        threshold = 0.90 

        # 2. Optimized Logic: Masking & Stacking
        # Create a mask for the upper triangle (k = 1 excludes the diagonal)
        mask = np.triu(np.ones(corr_matrix.shape, dtype = bool), k = 1)
                
        # Apply mask: Keep only upper triangle values, turn others to NaN
        # Stack to flatten into a Series, dropping NaNs automatically
        high_corr_pairs = corr_matrix.where(mask).stack()
                
        # Filter for absolute correlation greater than threshold
        high_corr_pairs = high_corr_pairs[high_corr_pairs.abs() > threshold]

        # 3. Correlation Heatmap (Hidden by Default)
        with st.expander("📊 View Correlation Matrix Details"):
            st.write("Correlation measures how two assets move in relation to each other.")
            st.write("- **1.0**: Perfect Positive Correlation (Move together)")
            st.write("- **0.0**: No Correlation")
            st.write("- **-1.0**: Perfect Negative Correlation (Move opposite)")
                    
            fig_corr = go.Figure(data = go.Heatmap(z = corr_matrix.values,
                                                   x = corr_matrix.columns,
                                                   y = corr_matrix.columns,
                                                   colorscale = 'RdBu',
                                                   zmid = 0, zmin = -1, zmax = 1,
                                                   text = corr_matrix.values.round(2),
                                                   texttemplate = "%{text}",
                                                   showscale = True))
            
            fig_corr.update_layout(height = 400,
                                   title = "Asset Correlation Matrix",
                                   yaxis = dict(autorange = "reversed"))
                    
            st.plotly_chart(fig_corr, use_container_width = True)

        # 3. Display Warning
        if not high_corr_pairs.empty:
            st.warning(f"⚠️ **Alert: High Correlation Detected!**")
            st.caption(f"Some selected assets behave very similarly (correlation > {threshold}). Holding both may not provide effective diversification.")
                    
            # high_corr_pairs is a MultiIndex Series: (Ticker1, Ticker2) -> Correlation
            for (ticker1, ticker2), score in high_corr_pairs.items():
                relation = "Positive" if score > 0 else "Negative"
                st.markdown(f"- **{ticker1}** & **{ticker2}**: {score:.2f} ({relation})")

        # --- Final Allocation Output ---
        st.subheader("Optimal Asset Allocation")
                    
        allocation_df = pd.DataFrame({"Ticker": cols, "Weight": opt_weights})
        allocation_df = allocation_df.sort_values(by = "Weight", ascending = False)
        allocation_df['Weight'] = allocation_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
                    
        st.table(allocation_df.set_index('Ticker'))

# ==========================================
# MODULE 3: PORTFOLIO REBALANCER
# ==========================================

elif page == "Portfolio Rebalancer":
    st.header("Portfolio Rebalancing Assistant")
    st.caption("Calculate trades to align your portfolio with target allocations.")

    # --- 1. Global Inputs (Cash) ---
    # Scheme B: Independent Cash Input
    current_cash = st.number_input("Current Cash Balance ($)", 
                                   min_value = 0.0, 
                                   value = 10000.0, 
                                   step = 100.0,
                                   help = "Enter the amount of uninvested cash you currently hold.")

    st.divider()

    col_input, col_output = st.columns([1, 1], gap = "medium")

    # --- 2. Input Section (Left) ---
    with col_input:
        st.subheader("Current Holdings")
        
        # Initialize default data structure for the editor
        if 'rebalance_data' not in st.session_state:
            default_data = {
                "Ticker": ["VTI", "VXUS", "BND"],
                "Shares": [50, 30, 20],
                "Target (%)": [60.0, 30.0, 10.0]
            }
            st.session_state['rebalance_data'] = pd.DataFrame(default_data)

        # Data Editor (Excel-like input)
        input_df = st.data_editor(st.session_state['rebalance_data'], 
                                  num_rows = "dynamic", 
                                  use_container_width = True,
                                  column_config = {
                                      "Ticker": st.column_config.TextColumn("Ticker", required = True),
                                      "Shares": st.column_config.NumberColumn("Shares", min_value = 0, step = 1, format = "%d"),
                                      "Target (%)": st.column_config.NumberColumn("Target %", min_value = 0, max_value = 100, step = 0.1, format = "%.1f%%")
                                  })
        
        # Save changes to session state to prevent data loss on rerun
        st.session_state['rebalance_data'] = input_df

        # Action Button
        st.write("")
        calculate_btn = st.button("Calculate Rebalancing", type = "primary", use_container_width = True)

    # --- 3. Calculation Logic & Output (Right) ---
    with col_output:
        st.subheader("Rebalancing Plan")

        if calculate_btn:
            # A. Validation
            valid_rows = input_df[input_df['Ticker'].notna() & (input_df['Ticker'] != "")]
            
            if valid_rows.empty:
                st.warning("Please enter at least one valid ticker.")
            
            elif valid_rows['Target (%)'].sum() > 100.1: # Allow small float error
                st.error(f"Total Target Allocation ({valid_rows['Target (%)'].sum():.1f}%) exceeds 100%. Please adjust.")
            
            else:
                with st.spinner("Fetching latest prices..."):
                    
                    # B. Fetch Data
                    tickers = valid_rows['Ticker'].str.upper().tolist()
                    # Fetch 5 days to ensure we get the last closing price even on weekends/holidays
                    market_data = get_stock_data(tickers, period = "5d") 

                    if market_data is None or market_data.empty:
                        st.error("Failed to fetch stock data. Please check your tickers.")
                    
                    else:
                        try:
                            # Use helper function to extract price data consistently
                            price_data = extract_price_data(market_data, prefer_adj_close=True)
                            
                            if price_data is None or price_data.empty:
                                st.error("Failed to extract price data. Please check your tickers.")
                            else:
                                # Get last valid prices (handles both single and multiple tickers)
                                last_prices = price_data.iloc[-1]
                                current_prices = last_prices.to_dict()

                            # C. Core Math (Rebalancing)
                            results = []
                            total_equity = current_cash
                            
                            # 1. Calculate Total Portfolio Value first
                            for index, row in valid_rows.iterrows():
                                ticker = row['Ticker'].upper()
                                shares = row['Shares']
                                price = current_prices.get(ticker, 0.0)
                                
                                if price == 0.0:
                                    st.warning(f"Could not find price for {ticker}. Skipping.")
                                    continue
                                
                                position_value = shares * price
                                total_equity += position_value
                            
                            # 2. Calculate New Allocation
                            projected_cash = total_equity # Start with total, subtract as we 'buy'
                            
                            for index, row in valid_rows.iterrows():
                                ticker = row['Ticker'].upper()
                                current_shares = row['Shares']
                                target_pct = row['Target (%)'] / 100.0
                                price = current_prices.get(ticker, 0)
                                
                                if price > 0:
                                    # Target Value for this stock
                                    target_value = total_equity * target_pct
                                    
                                    # Calculate New Shares (Floor division to avoid fractional shares)
                                    new_shares = int(np.floor(target_value / price))
                                    
                                    # Calculate Trades
                                    trade_shares = new_shares - current_shares
                                    
                                    # Final Value for this stock
                                    final_value = new_shares * price
                                    projected_cash -= final_value # Deduct cost from total pool
                                    
                                    # Actual achieved weight (may differ slightly due to rounding)
                                    actual_weight = (final_value / total_equity) * 100
                                    
                                    results.append({
                                        "Ticker": ticker,
                                        "New Shares": new_shares,
                                        "Trade (+/-)": trade_shares, # Key output
                                        "Value ($)": final_value,
                                        "Actual %": actual_weight
                                    })

                            # D. Construct Results DataFrame
                            res_df = pd.DataFrame(results)
                            
                            # Formatting for display
                            display_df = res_df.copy()
                            display_df['Value ($)'] = display_df['Value ($)'].apply(lambda x: f"${x:,.0f}")
                            display_df['Actual %'] = display_df['Actual %'].apply(lambda x: f"{x:.1f}%")
                            display_df['Trade (+/-)'] = display_df['Trade (+/-)'].apply(lambda x: f"+{x}" if x > 0 else f"{x}")

                            # Show Main Table
                            st.dataframe(display_df, hide_index = True, use_container_width = True)

                            # Show Cash Summary
                            # The remaining cash after buying integer shares
                            cash_pct = (projected_cash / total_equity) * 100
                            
                            st.info(f"""
                                    **Result Summary:**
                                    - **Total Portfolio Value:** ${total_equity:,.2f}
                                    - **Remaining Cash:** ${projected_cash:,.2f} ({cash_pct:.1f}%)
                                    """)

                            if projected_cash < 0:
                                st.error("Warning: Negative cash balance! Please reduce target percentages or add cash.")

                        except Exception as e:
                            st.error(f"An error occurred during calculation: {e}")

        else:
            st.info("👈 Enter your holdings and targets on the left, then click Calculate.")

# ==========================================
#  Footer & Disclaimer 
# ==========================================

st.write("") 
st.write("")
st.divider() 

with st.container():
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        <p><strong>Disclaimer:</strong> This application is for <strong>educational and informational purposes</strong> only. 
        The information presented does not constitute financial advice or recommendation to buy or sell any securities.
        All models are based on historical data and statistical assumptions, which do not guarantee future performance.</p>
        <p>2025 Jason Huang | Data Source: Yahoo Finance | Built with Streamlit, Python & Gemini</p>
        </div>
        """, 
        unsafe_allow_html = True)
