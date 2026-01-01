"""
Financial Analysis & Optimization Tool

A Streamlit application for stock price forecasting, portfolio optimization, and rebalancing.
Uses Monte Carlo simulation and Modern Portfolio Theory for financial analysis.

Author: Jason Huang
Year: 2025
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go 
 
import streamlit as st
import yfinance as yf
import utils

# ==========================================
# Configuration Constants
# ==========================================

# API Configuration
CACHE_TTL_SECONDS = 3600  # 1 hour cache for stock data
DEFAULT_DATA_PERIOD = "1y"  # Default period for stock forecaster
OPTIMIZER_DATA_PERIOD = "3y"  # Period for portfolio optimizer
REBALANCER_DATA_PERIOD = "5d"  # Period for rebalancer (to get latest prices)

# Monte Carlo Simulation Defaults
DEFAULT_SIMULATIONS = 200
DEFAULT_TIME_HORIZON = 20  # Trading days
DEFAULT_RANDOM_SEED = 42
MAX_SIMULATIONS = 1000
MIN_SIMULATIONS = 100
MAX_TIME_HORIZON = 365
MIN_TIME_HORIZON = 5
MAX_LINES_TO_PLOT = 50  # Limit visualization lines for performance

# Portfolio Optimizer Defaults
DEFAULT_NUM_PORTFOLIOS = 5000
MIN_NUM_PORTFOLIOS = 1000
MAX_NUM_PORTFOLIOS = 10000
DEFAULT_RISK_FREE_RATE = 3.0  # Percentage
MIN_RISK_FREE_RATE = 0.0
MAX_RISK_FREE_RATE = 10.0
HIGH_CORRELATION_THRESHOLD = 0.90  # Alert if correlation > this value

# Portfolio Rebalancer Defaults
DEFAULT_CASH_BALANCE = 10000.0
ALLOCATION_TOLERANCE = 0.1  # Allow small float error in percentage sums
MAX_CASH_PERCENTAGE_WARNING = 5.0  # Warn if cash > 5% of portfolio

# Risk Analysis Constants
VAR_CONFIDENCE_LEVEL = 0.05  # 5th percentile for VaR (95% confidence)
MIN_VOLATILITY_FOR_SHARPE = 1e-10  # Avoid division by zero in Sharpe ratio

# ==========================================
# Application Configuration
# ==========================================

st.set_page_config(page_title = "Financial Analysis Tool", 
                   layout = "wide",
                   initial_sidebar_state = "collapsed")

# Inject custom CSS for enhanced styling
st.markdown("""
            <style>
                /* Metric styling */
                div[data-testid = "stMetricValue"] {
                    font-size: 24px;
                    font-weight: 600;
                }
                
                /* Header styling */
                h1 {
                    color: #1f77b4;
                }
                
                /* Button hover effects */
                .stButton > button {
                    transition: all 0.3s ease;
                }
                
                .stButton > button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                }
                
                /* Info boxes */
                .stInfo {
                    border-left: 4px solid #1f77b4;
                }
                
                /* Success indicators */
                .success-box {
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px 0;
                }
            </style>
            """, 
            unsafe_allow_html = True)

# ==========================================
# Main Page Header
# ==========================================

st.title("📊 Financial Analysis Tool")
st.caption("Advanced Financial Modeling & Optimization | Powered by Modern Portfolio Theory & Monte Carlo Simulation")

# Page Navigation with better labels
page = st.radio("Select Tool:", 
                ["📈 Stock Price Forecaster", "⚖️ Portfolio Optimizer", "🔄 Portfolio Rebalancer"], 
                horizontal = True,
                label_visibility = "collapsed")

st.markdown("---")

# ==========================================
# Load Data Using Yahoo Finance API
# ==========================================



# ==========================================
# MODULE 1: STOCK PRICE FORECASTER
# ==========================================

if page == "📈 Stock Price Forecaster":
    
    st.header("📈 Stock Price Prediction")
    st.markdown("**Forecast future stock prices using Monte Carlo simulation based on historical volatility**")
    
    # --- Sidebar Settings ---  
    st.sidebar.header("⚙️ Simulation Parameters")
    
    with st.sidebar.expander("💡 Quick Tips", expanded = False):
        st.markdown("""
        - **Time Horizon**: Longer periods = more uncertainty
        - **Simulations**: More simulations = more accurate but slower
        """)
    
    time_horizon = st.sidebar.slider("Time Horizon (Trading Days)", 
                                     MIN_TIME_HORIZON, MAX_TIME_HORIZON, DEFAULT_TIME_HORIZON,
                                     help = "Number of trading days into the future for prediction. Typical: 20 days = ~1 month, 252 days = ~1 year")
    
    simulations = st.sidebar.slider("Number of Simulations", 
                                    MIN_SIMULATIONS, MAX_SIMULATIONS, DEFAULT_SIMULATIONS,
                                    help = "More simulations = more accurate results, but slower speed. Recommended: 200-500 for balance")
    
    seed = st.sidebar.number_input("Random Seed", 
                                   value = DEFAULT_RANDOM_SEED, 
                                   min_value = 0,
                                   step = 1,
                                   format = "%d",
                                   help = "Fix the random numbers for reproducible results. Change to get different scenarios.")
    
    # --- Input Section ---
    col1, col2 = st.columns([4, 1]) 
    
    with col1:
        ticker = st.text_input("Enter Stock Ticker", 
                               value = "VOO", 
                               placeholder="e.g., AAPL, GOOGL, MSFT, VOO",
                               help = "Enter a valid stock ticker symbol. Examples: VOO, AAPL, GOOGL")
    with col2:
        st.write("") 
        st.write("") 
        start_sim = st.button("🚀 Start Simulation", 
                              type = "primary", 
                              use_container_width = True)
    
    # --- Simulation Logic ---
    if start_sim:
        # Input validation
        if not ticker or not ticker.strip():
            st.error("❌ **Please enter a stock ticker symbol.**")
            st.info("💡 **Tip:** Try popular tickers like AAPL, GOOGL, MSFT, or VOO")
        else:
            ticker = ticker.strip().upper()
            with st.spinner('🔄 Running Monte Carlo Simulation... This may take a few seconds.'):
                np.random.seed(int(round(seed)))
                stock_data = utils.get_stock_data(ticker, period = DEFAULT_DATA_PERIOD)

                if stock_data is None or stock_data.empty:
                    st.error(f"❌ **Ticker '{ticker}' not found or API unavailable.**")
                    st.info(f"""
                    **Troubleshooting:**
                    - Check if the ticker symbol is correct (e.g., AAPL not Apple)
                    - Try a different ticker (e.g., VOO, SPY, MSFT)
                    - The API might be temporarily unavailable - please try again in a moment
                    """)
                    
                else:
                    # Fetch full name using cached function
                    stock_info = utils.get_stock_info(ticker)
                    stock_name = stock_info.get('longName', ticker) if stock_info else ticker

                    # Data Preprocessing using helper function
                    price_data = utils.extract_price_data(stock_data, prefer_adj_close=True)
                    
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
                    sim_results = utils.run_monte_carlo_simulation(last_price, log_returns, time_horizon, simulations)

                    # --- SAVE TO SESSION STATE ---
                    st.session_state['forecast_results'] = {
                        'simulation_df': sim_results['simulation_df'],
                        'last_price': last_price,
                        'stock_name': stock_name,      
                        'price_change': price_change,  
                        'pct_change': pct_change,    
                        'expected_price': sim_results['expected_price'],
                        'median_price': sim_results['median_price'],
                        'worst_case': sim_results['worst_case'],
                        'cvar_95': sim_results['cvar_95'],
                        'prob_loss': sim_results['prob_loss'],
                        'ticker': ticker,
                        'time_horizon': time_horizon, 
                        'simulations': simulations
                    }
                    
                    # Success message
                    st.success(f"✅ **Simulation completed successfully!** Analyzed {simulations} scenarios for {ticker} over {time_horizon} trading days.")

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
        col_header1, col_header2 = st.columns([1, 1]) # Left gets more space

        with col_header1:
            st.markdown(f"<h1 style='margin-bottom:0px;'>{saved_ticker}</h1>", unsafe_allow_html = True)
            st.markdown(f"<p style='font-size: 1.1rem; margin-top: 0px; margin-bottom: 0px;'>{saved_name}</p>", unsafe_allow_html = True) # Larger font for full name
            st.write("")  # Add spacing to align with price metric
        
        with col_header2:
            st.markdown(f"<h1 style='margin-bottom:0px;'>Current Price: ${last_price:.2f}</h1>", unsafe_allow_html = True)
            # Use same color scheme as st.metric() delta_color="normal" (green for positive, red for negative)
            # Streamlit metric colors: positive=#28a745, negative=#dc3545
            color = "#28a745" if saved_pct >= 0 else "#dc3545"
            st.markdown(f"<p style='color: {color}; font-size: 1.1rem; margin-top: 0px; font-weight: 500;'>{saved_change:+.2f} ({saved_pct:+.2f}%)</p>", unsafe_allow_html = True)
            
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
                                     opacity = 0.3,
                                     line = dict(width = 1, color = '#636EFA'),
                                     showlegend = False,
                                     hoverinfo = 'skip' ))
    
        # Add Mean Expectation Line (precomputed from full array)
        if 'Mean' in simulation_df.columns:
            fig.add_trace(go.Scatter(x = simulation_df.index,
                                     y = simulation_df['Mean'],
                                     mode = 'lines',
                                     name = '📊 Expected Average',
                                     line = dict(color = '#EF553B', width = 3),
                                     opacity = 1.0,
                                     hovertemplate = 'Day %{x}<br>Price: $%{y:.2f}<extra></extra>'))
        
        # Add current price reference line (as a trace so it appears in legend)
        fig.add_trace(go.Scatter(x = simulation_df.index,
                                 y = [last_price] * len(simulation_df.index),
                                 mode = 'lines',
                                 name = '💰 Current Price',
                                 line = dict(color = 'green', width = 2, dash = 'dash'),
                                 opacity = 1.0,
                                 hovertemplate = f'Current Price: ${last_price:.2f}<extra></extra>'))
                
        # Layout setting with better styling
        fig.update_layout(
            title = dict(
                text = f"📈 {saved_sims} Monte Carlo Simulation Scenarios for {saved_ticker}",
                font = dict(size = 18)
            ),
            xaxis_title = "Trading Days into Future",
            yaxis_title = "Price (USD)",
            xaxis = dict(range = [0, saved_horizon]),
            hovermode = "x unified",
            template = "plotly_white",
            height = 500,
            showlegend = True,
            legend = dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Render
        st.plotly_chart(fig, use_container_width = True)
                
        # Guide: Interpretation
        with st.expander("ℹ️ How to interpret this chart?"):
            st.write(
                """
                This chart shows possible future price paths based on historical volatility.
                - **Red Line**: The average expected price trend.
                - **Green Dashed Line**: The current price reference line for comparison.
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
        st.subheader("📊 Risk Analysis & Forecast Summary")
        
        # ROI Calculation
        expected_pct = (expected_price - last_price) / last_price * 100
        median_pct = (median_price   - last_price) / last_price * 100
        worst_pct = (worst_case - last_price) / last_price * 100
        cvar_pct = (cvar_95 - last_price) / last_price * 100

        # Setup Layout (2 Columns by 2 Rows)
        col1, col2 = st.columns(2)

        # Display Metrics with color-coded deltas
        col1.metric("📈 Expected Price (Average)", 
                    f"${expected_price:.2f}", 
                    f"{expected_pct:+.2f}%",
                    delta_color = "normal" if expected_pct >= 0 else "inverse")

        col2.metric("📊 Median Price (50th Percentile)",
                    f"${median_price:.2f}", 
                    f"{median_pct:+.2f}%",
                    delta_color = "normal" if median_pct >= 0 else "inverse")

        col3, col4 = st.columns(2)
                
        col3.metric("⚠️ Value at Risk (95% Confidence)",
                    f"${worst_case:.2f}", 
                    f"{worst_pct:+.2f}%",
                    delta_color = "normal",
                    help = "5th Percentile outcome. Indicates a 95% probability that price remains above this level.")
                
        col4.metric("🔻 CVaR / Expected Shortfall (95%)",
                    f"${cvar_95:.2f}",
                    f"{cvar_pct:+.2f}%",
                    delta_color = "normal",
                    help = "Average terminal price within the worst 5% outcomes. This describes tail severity beyond VaR.")
        
        # Risk indicator below CVaR
        prob_loss_pct = prob_loss*100
        loss_color = "🔴" if prob_loss_pct > 50 else "🟡" if prob_loss_pct > 30 else "🟢"

        col5, col6 = st.columns([1, 1])

        # Probability of Loss metric
        col5.metric(f"{loss_color} Probability of Loss",
                  f"{prob_loss_pct:.1f}%",
                  help = "Share of simulations where the terminal price finishes below the current price.")
        
        with col6:
            # Use nested columns to limit the width of the warning message
            col6a, col6b = st.columns([1, 1])
            with col6a:
                if prob_loss_pct < 30:
                    st.success("✅ Low risk of loss")
                elif prob_loss_pct < 50:
                    st.warning("⚠️ Moderate risk of loss")
                else:
                    st.error("🔴 High risk of loss")

# ==========================================
# MODULE 2: PORTFOLIO OPTIMIZER (MPT)
# ==========================================

elif page == "⚖️ Portfolio Optimizer":
    st.header("⚖️ Efficient Frontier (Modern Portfolio Theory)")
    st.markdown("**Optimize your portfolio allocation to maximize returns while minimizing risk**")
    
    # --- Sidebar Settings ---  
    st.sidebar.header("⚙️ Optimization Settings")
    
    with st.sidebar.expander("💡 Quick Tips", expanded=False):
        st.markdown("""
        - **Diversification**: Mix stocks from different sectors/regions
        - **Risk-Free Rate**: ~2-5% (US Treasury rate)
        - **More Simulations**: Better accuracy but slower
        """)
    
    num_portfolios = st.sidebar.slider("Number of Simulations", 
                                       MIN_NUM_PORTFOLIOS, MAX_NUM_PORTFOLIOS, DEFAULT_NUM_PORTFOLIOS,
                                       help = "Higher simulations produce a more accurate Efficient Frontier. Recommended: 3000-5000 for balance")
    
    # Risk-Free Rate Input
    risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate (%)",
                                                   value = DEFAULT_RISK_FREE_RATE,
                                                   min_value = MIN_RISK_FREE_RATE,
                                                   max_value = MAX_RISK_FREE_RATE,
                                                   step = 0.1,
                                                   help = "Current annual risk-free rate (e.g., 3-month US Treasury Bill). Typical: 2-5%")
    
    # Convert to decimal
    risk_free_rate = risk_free_rate_input / 100

    seed = st.sidebar.number_input("Random Seed", 
                                   value = DEFAULT_RANDOM_SEED, 
                                   min_value = 0,
                                   step = 1,
                                   format = "%d",
                                   help = "Fix the random numbers for reproducible results. Change to get different optimization results.")

    # --- Input Section ---  
    col_input, col_btn = st.columns([4, 1]) 
    
    with col_input:
        tickers_input = st.text_input("Enter Stock Tickers (Comma Separated)", 
                                     value = "VTI, VEA, VNQ",
                                     placeholder="e.g., VTI, VEA, VNQ, BND",
                                     help = "Enter at least 2 tickers separated by commas. Mix different asset classes for better diversification (e.g., stocks, bonds, real estate).")
        
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        tickers = list(set(tickers))

    with col_btn:
        st.write("") 
        st.write("") 
        start_opt = st.button("🚀 Optimize", 
                              type = "primary", 
                              use_container_width = True)

    # --- Optimization Logic ---
    if start_opt:
        
        # --- 1. Early Validation ---
        # Check before making API calls to save time
        if len(tickers) < 2:
            st.error("❌ **At least 2 valid tickers are required for portfolio optimization.**")
            st.info("💡 **Tip:** Enter multiple tickers separated by commas (e.g., VTI, VEA, VNQ)")
            st.stop()
            
        with st.spinner('🔄 Calculating Efficient Frontier... This may take 10-30 seconds.'):
            np.random.seed(int(round(seed)))
            # Fetch data for specified period to calculate correlation matrix
            raw_data = utils.get_stock_data(tickers, period = OPTIMIZER_DATA_PERIOD)

            # Check if API returned any data
            if raw_data is None or raw_data.empty:
                st.error("Error: No data found. Please check your tickers.")
                st.stop()
                
            # --- 2. Data Cleaning & Selection ---
            try:
                # Use helper function to extract price data consistently
                data = utils.extract_price_data(raw_data, prefer_adj_close=True)
                
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
                # --- MPT Calculations & Simulation ---
                opt_data = utils.optimize_portfolio(data, risk_free_rate, num_portfolios)
                
                # --- SAVE TO SESSION STATE ---
                st.session_state['mpt_results'] = {'results': opt_data['results'],
                                                   'opt_std': opt_data['opt_std'],
                                                   'opt_ret': opt_data['opt_ret'],
                                                   'opt_weights': opt_data['opt_weights'],
                                                   'tickers': opt_data['tickers'],
                                                   'returns': opt_data['returns'],
                                                   'rf_rate': risk_free_rate}
                
                # Success message
                optimal_sharpe = (opt_data['opt_ret'] - risk_free_rate) / opt_data['opt_std']
                st.success(f"✅ **Optimization completed!** Analyzed {num_portfolios} portfolio combinations. Optimal Sharpe Ratio: {optimal_sharpe:.2f}")

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

        optimal_sharpe = (opt_ret - saved_rf) / opt_std
        fig.update_layout(
            title = dict(
                text = f"📊 Risk vs. Return Analysis (Risk-Free Rate: {saved_rf*100:.1f}%) | Optimal Sharpe: {optimal_sharpe:.2f}",
                font = dict(size = 16)
            ),
            xaxis_title = "Volatility (Annualized Std Dev)",
            yaxis_title = "Expected Annual Return",
            template = "plotly_white",
            height = 600,
            legend = dict(yanchor = "top", y = 0.99,
                        xanchor = "right", x = 0.99,
                        bgcolor = "rgba(255,255,255,0.8)",
                        bordercolor = "gray",
                        borderwidth = 1))
                
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
        threshold = HIGH_CORRELATION_THRESHOLD 

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
        st.divider()
        st.subheader("💼 Optimal Asset Allocation")
        
        # Calculate optimal portfolio metrics
        optimal_sharpe = (opt_ret - saved_rf) / opt_std
        optimal_return_pct = opt_ret * 100
        optimal_vol_pct = opt_std * 100
        
        # Display key metrics
        col_met1, col_met2, col_met3 = st.columns(3)
        col_met1.metric("📈 Expected Return", f"{optimal_return_pct:.2f}%")
        col_met2.metric("📊 Volatility", f"{optimal_vol_pct:.2f}%")
        col_met3.metric("⭐ Sharpe Ratio", f"{optimal_sharpe:.2f}")
                    
        allocation_df = pd.DataFrame({"Ticker": cols, "Weight": opt_weights})
        allocation_df = allocation_df.sort_values(by = "Weight", ascending = False)
        allocation_df['Weight'] = allocation_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
        
        # Get ticker with highest weight BEFORE setting index
        max_weight_ticker = allocation_df.iloc[0]['Ticker']
        max_weight = float(allocation_df.iloc[0]['Weight'].replace('%', ''))
        
        # Add visual bars for weights
        st.markdown("**Allocation Breakdown:**")
        st.dataframe(allocation_df.set_index('Ticker'), use_container_width=True)
        
        # Interpretation
        if max_weight > 50:
            st.info(f"💡 **Note:** {max_weight_ticker} has a high allocation ({allocation_df.iloc[0]['Weight']}). Consider if this matches your risk tolerance.")

# ==========================================
# MODULE 3: PORTFOLIO REBALANCER
# ==========================================

elif page == "🔄 Portfolio Rebalancer":
    st.header("🔄 Portfolio Rebalancing Assistant")
    st.markdown("**Calculate trades needed to align your portfolio with target allocations**")

    # --- 1. Global Inputs (Cash) ---
    with st.expander("💡 How to use the Rebalancer", expanded=False):
        st.markdown("""
        1. **Enter your current cash balance** (uninvested money)
        2. **Add your current holdings** (ticker, number of shares, target %)
        3. **Click Calculate** to see the rebalancing plan
        4. **Review the trades** needed to reach your target allocation
        """)
    
    # Initialize or retrieve cash from session state
    if 'rebalance_cash' not in st.session_state:
        st.session_state['rebalance_cash'] = DEFAULT_CASH_BALANCE
    
    current_cash = st.number_input("💰 Current Cash Balance ($)", 
                                   min_value = 0.0, 
                                   value = st.session_state['rebalance_cash'], 
                                   step = 100.0,
                                   help = "Enter the amount of uninvested cash you currently hold. This will be used to purchase additional shares.",
                                   key = "cash_input")
    
    # Save cash to session state
    st.session_state['rebalance_cash'] = current_cash

    st.divider()

    col_input, col_output = st.columns([1, 1], gap = "medium")

    # --- 2. Input Section (Left) ---
    with col_input:
        st.subheader("📋 Current Holdings")
        
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
                                  },
                                  hide_index = True)
        
        # Save changes to session state to prevent data loss on rerun
        st.session_state['rebalance_data'] = input_df

        # Action Button
        st.write("")
        calculate_btn = st.button("🚀 Calculate Rebalancing", type = "primary", use_container_width = True)
        
        # Show total target allocation
        if 'rebalance_data' in st.session_state:
            total_target = st.session_state['rebalance_data']['Target (%)'].sum()
            if total_target > 0:
                if total_target > 100.1:
                    st.error(f"⚠️ Total allocation: {total_target:.1f}% (exceeds 100%)")
                elif total_target < 99.9:
                    st.warning(f"ℹ️ Total allocation: {total_target:.1f}% (less than 100%)")
                else:
                    st.success(f"✅ Total allocation: {total_target:.1f}%")

    # --- 3. Calculation Logic & Output (Right) ---
    with col_output:
        st.subheader("📊 Rebalancing Plan")

        if calculate_btn:
            # A. Validation
            valid_rows = input_df[input_df['Ticker'].notna() & (input_df['Ticker'] != "")]
            
            if valid_rows.empty:
                st.warning("⚠️ **Please enter at least one valid ticker.**")
                st.info("💡 **Tip:** Add rows using the + button and enter ticker symbols (e.g., VTI, AAPL)")
            
            elif valid_rows['Target (%)'].sum() > 100.0 + ALLOCATION_TOLERANCE: # Allow small float error
                st.error(f"❌ **Total Target Allocation ({valid_rows['Target (%)'].sum():.1f}%) exceeds 100%.**")
                st.info("💡 **Tip:** Reduce target percentages so they sum to 100% or less")
            
            else:
                with st.spinner("Fetching latest prices..."):
                    
                    # B. Fetch Data
                    tickers = valid_rows['Ticker'].str.upper().tolist()
                    # Fetch recent days to ensure we get the last closing price even on weekends/holidays
                    market_data = utils.get_stock_data(tickers, period = REBALANCER_DATA_PERIOD) 

                    if market_data is None or market_data.empty:
                        st.error("Failed to fetch stock data. Please check your tickers.")
                    
                    else:
                        try:
                            # Use helper function to extract price data consistently
                            price_data = utils.extract_price_data(market_data, prefer_adj_close=True)
                            
                            if price_data is None or price_data.empty:
                                st.error("Failed to extract price data. Please check your tickers.")
                            else:
                                # Get last valid prices (handles both single and multiple tickers)
                                last_prices = price_data.iloc[-1]
                                current_prices = last_prices.to_dict()

                                # Optional: Warn for missing prices
                                for index, row in valid_rows.iterrows():
                                    ticker = row['Ticker'].upper()
                                    if current_prices.get(ticker, 0.0) == 0.0:
                                         st.warning(f"Could not find price for {ticker}. Skipping in calculation.")

                                # C. Core Math (Rebalancing)
                                plan = utils.calculate_rebalancing_plan(current_cash, valid_rows, current_prices)
                                
                                if 'error' in plan:
                                    st.error(f"❌ **Error:** {plan['error']}")
                                else:
                                    # --- SAVE TO SESSION STATE ---
                                    st.session_state['rebalance_results'] = {
                                        'results_df': plan['results_df'],
                                        'total_equity': plan['total_equity'],
                                        'projected_cash': plan['projected_cash'],
                                        'current_prices': current_prices,
                                        'current_cash': current_cash
                                    }
                                    
                                    # Formatting for display
                                    display_df = res_df.copy()
                                    display_df['Value ($)'] = display_df['Value ($)'].apply(lambda x: f"${x:,.0f}")
                                    display_df['Actual %'] = display_df['Actual %'].apply(lambda x: f"{x:.1f}%")
                                    display_df['Trade (+/-)'] = display_df['Trade (+/-)'].apply(lambda x: f"+{x}" if x > 0 else f"{x}")

                                    # Show Main Table (centered)
                                    st.dataframe(display_df, hide_index = True, use_container_width = True)
                                    
                                    # Success message (moved below table)
                                    st.success("✅ **Rebalancing plan calculated successfully!**")

                                    # Show Cash Summary
                                    # The remaining cash after buying integer shares
                                    cash_pct = (projected_cash / total_equity) * 100
                                    
                                    st.info(f"""
                                            **💰 Portfolio Summary:**
                                            - **Total Portfolio Value:** ${total_equity:,.2f}
                                            - **Remaining Cash:** ${projected_cash:,.2f} ({cash_pct:.1f}%)
                                            """)

                                    if projected_cash < 0:
                                        st.error("❌ **Warning: Negative cash balance!** Please reduce target percentages or add more cash.")
                                    elif cash_pct > MAX_CASH_PERCENTAGE_WARNING:
                                        st.warning(f"ℹ️ **Note:** {cash_pct:.1f}% of portfolio remains in cash due to integer share constraints.")

                        except Exception as e:
                            st.error(f"An error occurred during calculation: {e}")

        # Display saved results if they exist (when user returns to this module)
        elif 'rebalance_results' in st.session_state:
            try:
                saved_results = st.session_state['rebalance_results']
                res_df = saved_results.get('results_df')
                total_equity = saved_results.get('total_equity')
                projected_cash = saved_results.get('projected_cash')
                
                # Validate that all required keys exist
                if res_df is None or total_equity is None or projected_cash is None:
                    raise KeyError("Missing required keys in saved results")
                
                # Formatting for display
                display_df = res_df.copy()
                display_df['Value ($)'] = display_df['Value ($)'].apply(lambda x: f"${x:,.0f}")
                display_df['Actual %'] = display_df['Actual %'].apply(lambda x: f"{x:.1f}%")
                display_df['Trade (+/-)'] = display_df['Trade (+/-)'].apply(lambda x: f"+{x}" if x > 0 else f"{x}")
                
                # Show Main Table (centered)
                st.dataframe(display_df, hide_index = True, use_container_width = True)
                
                # Info message (moved below table)
                st.info("💾 **Displaying previously calculated rebalancing plan.** Click Calculate Rebalancing to recalculate with current data.")

                # Show Cash Summary (with validation)
                if total_equity > 0:
                    cash_pct = (projected_cash / total_equity) * 100
                    
                    st.info(f"""
                            **💰 Portfolio Summary:**
                            - **Total Portfolio Value:** ${total_equity:,.2f}
                            - **Remaining Cash:** ${projected_cash:,.2f} ({cash_pct:.1f}%)
                            """)
                    
                    if projected_cash < 0:
                        st.error("❌ **Warning: Negative cash balance!** Please reduce target percentages or add more cash.")
                    elif cash_pct > MAX_CASH_PERCENTAGE_WARNING:
                        st.warning(f"ℹ️ **Note:** {cash_pct:.1f}% of portfolio remains in cash due to integer share constraints.")
                else:
                    st.error("❌ **Error: Invalid portfolio value in saved results.**")
            except (KeyError, AttributeError, TypeError) as e:
                st.error(f"❌ **Error loading saved results:** {str(e)}")
                st.info("💡 **Tip:** Please recalculate your rebalancing plan.")
                # Clear corrupted session state
                if 'rebalance_results' in st.session_state:
                    del st.session_state['rebalance_results']

        else:
            st.info("👈 **Enter your holdings and targets on the left, then click Calculate Rebalancing.**")
            st.markdown("""
            **Quick Start:**
            - Enter your current cash balance above
            - Add your stock holdings (ticker, shares, target %)
            - Make sure target percentages sum to 100%
            - Click Calculate to see your rebalancing plan
            """)

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
