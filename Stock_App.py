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

    This function handles both single-ticker requests (returning a standard DataFrame) 
    and multi-ticker batch requests (returning a multi-index DataFrame) to support both the Forecaster and Optimizer modules.

    Args:
        tickers (str or list[str]): A single ticker symbol (e.g., "AAPL") or a list of symbols (e.g., ["AAPL", "GOOG"]).
        period (str): The historical period to download (e.g., "1y", "3y", "max").

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing historical stock data (Open, High, Low, Close, Volume) if successful; 
                              None if an API error occurs.
    """
    
    try:
        if isinstance(tickers, list):
            # Batch download for Portfolio Optimizer
            data = yf.download(tickers, period = period, ignore_tz = True)
            return data
        else:
            # Single ticker fetch for Forecaster
            stock = yf.Ticker(tickers)
            return stock.history(period = period)
            
    except Exception as e:
        print(f"Yahoo Finance Error: {e}")
        return None

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

                # Try to fetch full name 
                try:
                    stock_info = yf.Ticker(ticker).info
                    stock_name = stock_info.get('longName', ticker)
                except:
                    stock_name = ticker # Fallback if API fails

                # Data Preprocessing
                if 'Adj Close' in stock_data.columns:
                    closing_prices = stock_data['Adj Close']
                elif 'Close' in stock_data.columns:
                    closing_prices = stock_data['Close']
                else:
                    st.error(f"Data Error: Closing price is missing for {ticker}.")
                    st.stop()

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
            
                simulation_df = pd.DataFrame(price_paths, columns = [f"Sim_{i}" for i in range(simulations)])

                # Compute Key Metrics
                end_prices = simulation_df.iloc[-1]

                # Center tendency
                expected_price = end_prices.mean()
                median_price = end_prices.median()
                
                # For worst case, we use a 95% CI              
                worst_case = float(end_prices.quantile(0.05, interpolation = "linear")) 

                # CVaR / Expected Shortfall (average of worst 5%)
                tail = end_prices[end_prices <= worst_case]
                cvar_95 = float(tail.mean()) if len(tail) > 0 else worst_case  # fallback safeguard
                
                # Probability of Loss
                prob_loss = float((end_prices < last_price).mean())  # 0~1

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
                
        # Performance Optimization: Limit rendered traces to prevent frontend lag
        max_lines_to_plot = 50
        columns_to_plot = list(simulation_df.columns[:min(saved_sims, max_lines_to_plot)])
                
        # Make sure the worst scenario is in the plot
        end_prices_local = simulation_df.iloc[-1]
        worst_col = (end_prices_local - worst_case).abs().idxmin()
                
        if worst_col not in columns_to_plot:
            columns_to_plot.append(worst_col)

        # Drawing the plot
        for col in columns_to_plot:
            fig.add_trace(go.Scatter(x = simulation_df.index,
                                     y = simulation_df[col],
                                     mode = 'lines', 
                                     opacity = 0.375,
                                     line = dict(width = 1),
                                     showlegend = False,
                                     hoverinfo = 'skip' ))
    
        # Add Mean Expectation Line
        mean_path = simulation_df.mean(axis = 1)
        fig.add_trace(go.Scatter(x = simulation_df.index,
                                 y = mean_path,
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
                # Prefer 'Adj Close', fallback to 'Close'
                if 'Adj Close' in raw_data.columns:
                    data = raw_data['Adj Close']
                        
                elif 'Close' in raw_data.columns:
                    data = raw_data['Close']
                    
                else:
                    st.error("Data Error: Neither 'Adj Close' nor 'Close' price columns found in API response.")
                    st.stop()
                    
                # --- 3. Post-Fetch Validation ---         
                # Drop columns that are entirely NaN (e.g., if a ticker is invalid but fetched)
                data = data.dropna(axis = 1, how = 'all')

                # Check if data is a Series (single stock) or DataFrame with < 2 columns
                if isinstance(data, pd.Series) or (isinstance(data, pd.DataFrame) and data.shape[1] < 2):
                    st.error("Error: Insufficient valid data. At least 2 valid stocks are needed to calculate correlation.")
                    st.stop()

                # Clean up MultiIndex if present 
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(1)

                # Final check for empty data after processing
                if data.empty:
                    st.error("Error: Processed data is empty.")
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
                            # Handle Data Structure (Single vs Multi-Index)
                            # We need a Series of Current Prices: {Ticker: Price}
                            current_prices = {}
                            
                            # Standardize column access (Prefer 'Adj Close', then 'Close')
                            price_col = 'Adj Close' if 'Adj Close' in market_data.columns else 'Close'
                            
                            if len(tickers) == 1:
                                # Single ticker returns a DataFrame with columns like [Open, Close...]
                                # Get the last valid price
                                last_valid_idx = market_data[price_col].last_valid_index()
                                price = float(market_data.loc[last_valid_idx, price_col])
                                current_prices[tickers[0]] = price
                            else:
                                # Batch returns MultiIndex columns: (Price_Type, Ticker)
                                # Extract just the price block
                                df_prices = market_data[price_col]
                                # Get last valid row
                                last_prices = df_prices.iloc[-1]
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
