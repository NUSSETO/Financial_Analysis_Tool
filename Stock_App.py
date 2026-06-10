"""
Financial Analysis & Optimization Tool

A Streamlit application for stock price forecasting, portfolio optimization,
rebalancing, historical stress testing, backtesting, and risk analysis.
Uses Monte Carlo simulation and Modern Portfolio Theory for financial analysis.

Author: Jason Huang
Year: 2026
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go 
from datetime import datetime, timedelta
 
import streamlit as st
import utils

# ==========================================
# Configuration Constants
# ==========================================

# API Configuration
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
st.caption("Market Data Extraction -> Statistical Analysis -> Robust Optimization")

# Page Navigation with better labels
page = st.radio("Select Tool:", 
                ["📈 Stock Price Forecaster", "⚖️ Portfolio Optimizer", "🔄 Portfolio Rebalancer",
                 "🏥 Stress Tester", "📊 Backtester", "📉 Risk Dashboard"], 
                horizontal = True,
                label_visibility = "collapsed")

st.markdown("---")

# ==========================================
# MODULE 1: STOCK PRICE FORECASTER
# ==========================================

if page == "📈 Stock Price Forecaster":
    
    st.header("📈 Stock Price Forecasting")
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
                    - Clear cache and rerun the app
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
                        'max_drawdown': sim_results['max_drawdown'],
                        'sharpe_ratio': sim_results['sharpe_ratio'],
                        'end_prices': sim_results['end_prices'],
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
        max_drawdown = res.get('max_drawdown', 0.0)
        sharpe_ratio = res.get('sharpe_ratio', 0.0)
        end_prices = res.get('end_prices', None)
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
        with st.expander("🧠 How does the simulation work? (Methodology)"):
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
                    delta_color = "normal")

        col2.metric("📊 Median Price (50th Percentile)",
                    f"${median_price:.2f}", 
                    f"{median_pct:+.2f}%",
                    delta_color = "normal")

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
        
        # --- New Risk Metrics: Max Drawdown & Sharpe ---
        col5, col6 = st.columns(2)
        
        col5.metric("📉 Max Drawdown (Worst Path)",
                    f"{max_drawdown*100:.1f}%",
                    help = "Worst peak-to-trough drop observed across all simulated price paths. Measures the deepest possible 'dip' during the holding period.")
        
        sharpe_color = "🟢" if sharpe_ratio > 1.0 else "🟡" if sharpe_ratio > 0 else "🔴"
        col6.metric(f"{sharpe_color} Annualized Sharpe Ratio",
                    f"{sharpe_ratio:.2f}",
                    help = "Risk-adjusted return from the simulation (annualized). > 1.0 is good, > 2.0 is very good.")

        # Risk indicator: Probability of Loss
        prob_loss_pct = prob_loss*100
        loss_color = "🔴" if prob_loss_pct > 50 else "🟡" if prob_loss_pct > 30 else "🟢"

        col7, col8 = st.columns([1, 1])

        # Probability of Loss metric
        col7.metric(f"{loss_color} Probability of Loss",
                  f"{prob_loss_pct:.1f}%",
                  help = "Share of simulations where the terminal price finishes below the current price.")
        
        with col8:
            # Use nested columns to limit the width of the warning message
            col8a, col8b = st.columns([1, 1])
            with col8a:
                if prob_loss_pct < 30:
                    st.success("✅ Low risk of loss")
                elif prob_loss_pct < 50:
                    st.warning("⚠️ Moderate risk of loss")
                else:
                    st.error("🔴 High risk of loss")
        
        # --- Terminal Price Distribution Histogram ---
        if end_prices is not None:
            st.divider()
            st.subheader("📊 Terminal Price Distribution")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x = end_prices,
                nbinsx = 60,
                marker_color = '#636EFA',
                opacity = 0.75,
                name = 'Terminal Prices'
            ))
            
            # Add vertical reference lines
            fig_hist.add_vline(x = last_price, line_dash = 'dash', line_color = 'green',
                               annotation_text = f'Current: ${last_price:.2f}', annotation_position = 'top left')
            fig_hist.add_vline(x = expected_price, line_dash = 'solid', line_color = '#EF553B',
                               annotation_text = f'Expected: ${expected_price:.2f}', annotation_position = 'top right')
            fig_hist.add_vline(x = worst_case, line_dash = 'dot', line_color = '#FFA15A',
                               annotation_text = f'VaR 95%: ${worst_case:.2f}', annotation_position = 'bottom left')
            
            fig_hist.update_layout(
                xaxis_title = 'Price (USD)',
                yaxis_title = 'Frequency',
                template = 'plotly_white',
                height = 350,
                showlegend = False,
                bargap = 0.05
            )
            
            st.plotly_chart(fig_hist, use_container_width = True)
            
            with st.expander("ℹ️ How to interpret this histogram?"):
                st.markdown("""
                This shows the **distribution of simulated end-of-period prices** across all scenarios.
                - **Green dashed line**: Current price — prices to the left represent losses.
                - **Red solid line**: Expected (average) terminal price.
                - **Orange dotted line**: Value at Risk (VaR) — 95% of outcomes are above this price.
                - A **wider spread** indicates higher uncertainty/volatility.
                - A **right-skewed** distribution suggests upside potential.
                """)

# ==========================================
# MODULE 2: PORTFOLIO OPTIMIZER (MPT)
# ==========================================

elif page == "⚖️ Portfolio Optimizer":
    st.header("⚖️ Efficient Frontier (Modern Portfolio Theory)")
    st.markdown("**Optimize your portfolio allocation to maximize returns while minimizing risk**")
    
    # --- Sidebar Settings ---  
    st.sidebar.header("⚙️ Model Settings")

    st.sidebar.subheader("Optimization Parameters")
    
    num_portfolios = st.sidebar.slider("Monte Carlo Simulations", 
                                       MIN_NUM_PORTFOLIOS, MAX_NUM_PORTFOLIOS, DEFAULT_NUM_PORTFOLIOS,
                                       help = "Number of random portfolios simulated to map the Efficient Frontier.")
    
    risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate (%)",
                                                   value = DEFAULT_RISK_FREE_RATE,
                                                   min_value = MIN_RISK_FREE_RATE,
                                                   max_value = MAX_RISK_FREE_RATE,
                                                   step = 0.1,
                                                   help = "Annualized risk-free rate (e.g., 10-Year Treasury Yield).")
    
    risk_free_rate = risk_free_rate_input / 100

    seed = st.sidebar.number_input("Random Seed", 
                                   value = DEFAULT_RANDOM_SEED, 
                                   min_value = 0,
                                   step = 1,
                                   format = "%d",
                                   help = "Seed for reproducibility.")

    st.sidebar.subheader("Model Methodology")
    model_choice = st.sidebar.radio("Optimization Model",
                                    ["Robust (Ledoit-Wolf)", "Classic (Sample Covariance)",
                                     "Black-Litterman", "Risk Parity", "Minimum CVaR"],
                                    help = "Robust: Ledoit-Wolf shrinkage (recommended).\nClassic: Standard Sample Covariance.\nBlack-Litterman: Blend your own views with market equilibrium.\nRisk Parity: Equal risk contribution from each asset.\nMinimum CVaR: Minimize tail risk (Expected Shortfall).")



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
    
    # --- Black-Litterman Views Input ---
    views_dict = {}
    if model_choice == "Black-Litterman":
        with st.expander("💡 Enter Your Return Views", expanded=True):
            st.markdown("Enter your expected **annual returns** for specific assets. Leave blank for no view on that asset.")
            views_cols = st.columns(min(len(tickers), 4)) if tickers else []
            for i, ticker in enumerate(tickers):
                col_idx = i % len(views_cols) if views_cols else 0
                with views_cols[col_idx]:
                    view_val = st.number_input(f"{ticker} (%)", value=0.0, step=1.0, 
                                               format="%.1f", key=f"view_{ticker}")
                    if view_val != 0.0:
                        views_dict[ticker] = view_val / 100.0

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
                frontier_data = None
                model_label = ""
                extra_data = {}
                
                if model_choice == "Classic (Sample Covariance)":
                    opt_data = utils.optimize_portfolio(data, risk_free_rate, num_portfolios)
                    model_label = "Standard Mean-Variance"
                    classic_returns = data.pct_change()
                    classic_mean = classic_returns.mean().values * 252
                    classic_cov = classic_returns.cov().values * 252
                    frontier_data = utils.compute_efficient_frontier(classic_mean, classic_cov)
                    
                elif model_choice == "Robust (Ledoit-Wolf)":
                    opt_data = utils.optimize_portfolio_robust(data.pct_change(), risk_free_rate, num_portfolios)
                    model_label = "Robust (Ledoit-Wolf)"
                    from sklearn.covariance import LedoitWolf
                    clean_rets = data.pct_change().dropna()
                    robust_mean = clean_rets.mean().values * 252
                    lw = LedoitWolf()
                    lw.fit(clean_rets)
                    robust_cov = lw.covariance_ * 252
                    frontier_data = utils.compute_efficient_frontier(robust_mean, robust_cov)
                    
                elif model_choice == "Black-Litterman":
                    if not views_dict:
                        st.warning("⚠️ No views entered. Please enter expected returns for at least one asset.")
                        st.stop()
                    opt_data = utils.optimize_portfolio_black_litterman(
                        data.pct_change(), views_dict, risk_free_rate, num_portfolios=num_portfolios)
                    model_label = "Black-Litterman"
                    if opt_data is not None:
                        extra_data['adjusted_returns'] = opt_data.get('adjusted_returns')
                        extra_data['equilibrium_returns'] = opt_data.get('equilibrium_returns')
                    
                elif model_choice == "Risk Parity":
                    opt_data = utils.optimize_portfolio_risk_parity(data.pct_change(), num_portfolios)
                    model_label = "Risk Parity"
                    if opt_data is not None:
                        extra_data['risk_contributions'] = opt_data.get('risk_contributions')

                elif model_choice == "Minimum CVaR":
                    opt_data = utils.optimize_portfolio_min_cvar(data.pct_change(), risk_free_rate, num_portfolios)
                    model_label = "Minimum CVaR"
                    if opt_data is not None:
                        extra_data['cvar'] = opt_data.get('cvar')
                        extra_data['alpha'] = opt_data.get('alpha')

                if opt_data is None:
                    st.error("Optimization failed. Please check your data or try different parameters.")
                    st.stop()
                
                st.success(f"✅ **{model_label} Optimization Complete**")
                
                # --- SAVE TO SESSION STATE ---
                st.session_state['mpt_results'] = {'results': opt_data['results'],
                                                   'opt_std': opt_data['opt_std'],
                                                   'opt_ret': opt_data['opt_ret'],
                                                   'opt_weights': opt_data['opt_weights'],
                                                   'tickers': opt_data['tickers'],
                                                   'returns': opt_data['returns'],
                                                   'rf_rate': risk_free_rate,
                                                   'frontier': frontier_data,
                                                   'model_label': model_label,
                                                   'extra': extra_data}
                
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
        frontier = data_store.get('frontier', None)
                
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
        
        # Overlay Efficient Frontier curve
        if frontier is not None:
            fig.add_trace(go.Scatter(x = frontier['frontier_vols'],
                                     y = frontier['frontier_rets'],
                                     mode = 'lines',
                                     name = 'Efficient Frontier',
                                     line = dict(color = '#FF6692', width = 3, dash = 'solid'),
                                     hovertemplate = 'Vol: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>'))

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
            with st.expander("🧠 How does the optimization work?"):
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

        with st.expander("📖 Methodology & Model Details"):
            st.markdown(
                """
                ### 1. Classic Markowitz (Mean-Variance Optimization)
                - **Objective**: Minimize portfolio variance for a given expected return.
                - **Input**: Sample Covariance Matrix (calculated directly from historical returns).
                - **Pros**: The standard academic baseline. Easy to interpret.
                - **Cons**: Extremely sensitive to "noise" in historical data. Often maximizes estimation error, leading to extreme weights (e.g., 100% allocation to one asset).

                ### 2. Robust Optimization (Ledoit-Wolf + CVXPY)
                - **Objective**: Minimize portfolio variance using a *shrinkage* estimator and convex optimization.
                - **Input**: **Ledoit-Wolf Covariance Matrix**. This "shrinks" the noisy sample covariance towards a structured target (constant correlation), reducing estimation error.
                - **Solver**: Uses **CVXPY**, a professional-grade convex optimization library, ensuring mathematically precise global minima (unlike random search).
                - **Why it matters**: In practice, sample covariance matrices are noisy. Robust methods prevent the optimizer from "chasing noise," resulting in more stable and diversified portfolios that perform better out-of-sample.

                ### 3. Minimum CVaR (Tail-Risk Optimization)
                - **Objective**: Minimize the **Conditional Value-at-Risk (CVaR / Expected Shortfall)** — the average loss on the worst tail of days (default: worst 5%).
                - **Method**: The **Rockafellar-Uryasev** linear formulation, solved as a convex program with **CVXPY** using historical daily returns directly as loss scenarios.
                - **Why it matters**: Variance penalizes upside and downside equally and assumes symmetric, Gaussian-like returns. CVaR targets *only* the left tail and makes no distributional assumption, so it is better suited to fat-tailed, crash-prone markets.
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
        extra_store = data_store.get('extra', {})
        cvar_value = extra_store.get('cvar')
        if cvar_value is not None:
            alpha_level = extra_store.get('alpha', 0.95)
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            col_met1.metric("📈 Expected Return", f"{optimal_return_pct:.2f}%")
            col_met2.metric("📊 Volatility", f"{optimal_vol_pct:.2f}%")
            col_met3.metric("⭐ Sharpe Ratio", f"{optimal_sharpe:.2f}")
            col_met4.metric(f"🛡️ Daily CVaR ({int(alpha_level*100)}%)", f"{cvar_value*100:.2f}%",
                            help="Expected loss on the worst tail of days (Expected Shortfall). "
                                 "This model minimizes exactly this tail risk.")
        else:
            col_met1, col_met2, col_met3 = st.columns(3)
            col_met1.metric("📈 Expected Return", f"{optimal_return_pct:.2f}%")
            col_met2.metric("📊 Volatility", f"{optimal_vol_pct:.2f}%")
            col_met3.metric("⭐ Sharpe Ratio", f"{optimal_sharpe:.2f}")
                    
        allocation_df = pd.DataFrame({"Ticker": cols, "Weight": opt_weights})
        allocation_df = allocation_df.sort_values(by = "Weight", ascending = False)
        
        # Get ticker with highest weight BEFORE formatting
        max_weight_ticker = allocation_df.iloc[0]['Ticker']
        max_weight = allocation_df.iloc[0]['Weight'] * 100
        
        # Display table and pie chart side by side
        col_table, col_pie = st.columns([1, 1])
        
        with col_table:
            st.markdown("**Allocation Breakdown:**")
            display_alloc = allocation_df.copy()
            display_alloc['Weight'] = display_alloc['Weight'].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(display_alloc.set_index('Ticker'), use_container_width=True)
        
        with col_pie:
            # Filter out zero-weight assets for a clean pie chart
            pie_df = allocation_df[allocation_df['Weight'] > 0.001]
            fig_pie = go.Figure(data=[go.Pie(
                labels = pie_df['Ticker'],
                values = pie_df['Weight'],
                hole = 0.4,
                textinfo = 'label+percent',
                textposition = 'outside',
                marker = dict(colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                                        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']),
                hovertemplate = '%{label}: %{percent}<extra></extra>'
            )])
            fig_pie.update_layout(
                showlegend = False,
                height = 350,
                margin = dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        if max_weight > 50:
            st.info(f"💡 **Note:** {max_weight_ticker} has a high allocation ({max_weight:.2f}%). Consider if this matches your risk tolerance.")

# ==========================================
# MODULE 3: PORTFOLIO REBALANCER
# ==========================================

elif page == "🔄 Portfolio Rebalancer":

    # --- Helper: Display rebalancing results (avoids duplicating formatting logic) ---
    def _display_rebalance_results(res_df, total_equity, projected_cash):
        """Formats and displays the rebalancing results table and cash summary."""
        display_df = res_df.copy()
        display_df['Value ($)'] = display_df['Value ($)'].apply(lambda x: f"${x:,.0f}")
        display_df['Actual %'] = display_df['Actual %'].apply(lambda x: f"{x:.1f}%")
        display_df['Trade (+/-)'] = display_df['Trade (+/-)'].apply(lambda x: f"+{x}" if x > 0 else f"{x}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)

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
            st.error("❌ **Error: Invalid portfolio value.**")

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
        
        # Use a SEPARATE key for data storage (not the widget key)
        # Streamlit doesn't allow pre-initializing the widget's own key
        if 'holdings_data' not in st.session_state:
            default_data = {
                "Ticker": ["VTI", "VXUS", "BND"],
                "Shares": [50, 30, 20],
                "Target (%)": [60.0, 30.0, 10.0]
            }
            st.session_state['holdings_data'] = pd.DataFrame(default_data)
        
        # Data Editor - pass the data value, use key for widget state management
        # The key prevents the reset-on-edit issue by letting Streamlit track widget state
        input_df = st.data_editor(st.session_state['holdings_data'], 
                                  num_rows = "dynamic", 
                                  use_container_width = True,
                                  column_config = {
                                      "Ticker": st.column_config.TextColumn("Ticker", required = True),
                                      "Shares": st.column_config.NumberColumn("Shares", min_value = 0, step = 1, format = "%d"),
                                      "Target (%)": st.column_config.NumberColumn("Target %", min_value = 0, max_value = 100, step = 0.1, format = "%.1f%%")
                                  },
                                  hide_index = True,
                                  key = "holdings_editor")
        
        # No sync needed - widget state is managed internally via the key parameter
        # input_df contains the current edited data for use in calculations

        # Action Button
        st.write("")
        calculate_btn = st.button("🚀 Calculate Rebalancing", type = "primary", use_container_width = True)
        
        # Show total target allocation (use input_df which reflects current state)
        total_target = input_df['Target (%)'].sum()
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
            # Persist the current input data to session state (for cross-page persistence)
            st.session_state['holdings_data'] = input_df.copy()
            
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
                                    res_df = plan['results_df']
                                    total_equity = plan['total_equity']
                                    projected_cash = plan['projected_cash']

                                    # --- SAVE TO SESSION STATE ---
                                    st.session_state['rebalance_results'] = {
                                        'results_df': res_df,
                                        'total_equity': total_equity,
                                        'projected_cash': projected_cash,
                                        'current_prices': current_prices,
                                        'current_cash': current_cash
                                    }
                                    
                                    # Display results using helper
                                    _display_rebalance_results(res_df, total_equity, projected_cash)
                                    st.success("✅ **Rebalancing plan calculated successfully!**")

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
                
                # Display results using helper
                _display_rebalance_results(res_df, total_equity, projected_cash)
                st.info("💾 **Displaying previously calculated rebalancing plan.** Click Calculate Rebalancing to recalculate with current data.")
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
# MODULE 4: STRESS TESTER
# ==========================================

elif page == "🏥 Stress Tester":
    st.header("🏥 Historical Stress Testing")
    st.markdown("**Test how your portfolio would have survived major market crises**")

    with st.expander("💡 How it works", expanded=False):
        st.markdown("""
        Enter your portfolio tickers and weights, then see how it would have performed during:
        - **Dot-Com Crash** (2000-2002): Tech bubble burst
        - **Global Financial Crisis** (2007-2009): Subprime mortgage crisis
        - **COVID-19 Crash** (2020): Pandemic market shock
        - **2022 Bear Market**: Inflation & rate hikes
        """)

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        stress_tickers_input = st.text_input("Enter Tickers (Comma Separated)", 
                                              value="VTI, BND, VNQ",
                                              key="stress_tickers",
                                              help="Enter tickers for your portfolio")
        stress_tickers = [t.strip().upper() for t in stress_tickers_input.split(",") if t.strip()]

    with col_s2:
        st.write("")
        st.write("")
        run_stress = st.button("🏥 Run Stress Test", type="primary", use_container_width=True)

    # Weights input
    if stress_tickers:
        st.markdown("**Portfolio Weights:**")
        weight_cols = st.columns(min(len(stress_tickers), 6))
        stress_weights = []
        for i, ticker in enumerate(stress_tickers):
            with weight_cols[i % len(weight_cols)]:
                w = st.number_input(f"{ticker} (%)", value=round(100.0/len(stress_tickers), 1), 
                                    min_value=0.0, max_value=100.0, step=5.0, key=f"sw_{ticker}")
                stress_weights.append(w / 100.0)

    if run_stress and stress_tickers:
        total_w = sum(stress_weights)
        if abs(total_w - 1.0) > 0.05:
            st.error(f"⚠️ Weights sum to {total_w*100:.1f}%. They should sum to ~100%.")
        else:
            stress_weights_arr = np.array(stress_weights) / sum(stress_weights)
            
            with st.spinner("🔄 Fetching historical data for crisis periods... This may take a moment."):
                results = utils.run_stress_test(stress_tickers, stress_weights_arr)

            if results:
                st.session_state['stress_results'] = results
                st.success("✅ **Stress test complete!**")

    if 'stress_results' in st.session_state:
        results = st.session_state['stress_results']
        
        # Summary table
        st.divider()
        st.subheader("📊 Crisis Performance Summary")
        
        summary_rows = []
        for r in results:
            if 'error' in r:
                summary_rows.append({
                    "Crisis": r['crisis'],
                    "Total Return": "N/A",
                    "Max Drawdown": "N/A",
                    "Recovered": "N/A"
                })
            else:
                summary_rows.append({
                    "Crisis": r['crisis'],
                    "Period": r['period'],
                    "Total Return": f"{r['total_return']*100:.1f}%",
                    "Max Drawdown": f"{r['max_drawdown']*100:.1f}%",
                    "Days to Trough": r['trough_day'],
                    "Recovered (6mo)": "✅ Yes" if r['recovered'] else "❌ No"
                })
        
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
        
        # Individual crisis charts
        st.divider()
        st.subheader("📈 Cumulative Return During Crises")
        
        for r in results:
            if 'error' not in r and 'cum_returns' in r:
                fig_crisis = go.Figure()
                dates = r['dates'][:len(r['cum_returns'])]
                
                fig_crisis.add_trace(go.Scatter(
                    x = dates, y = r['cum_returns'],
                    mode = 'lines', name = 'Portfolio',
                    line = dict(color='#636EFA', width=2)
                ))
                
                fig_crisis.add_hline(y=1.0, line_dash='dash', line_color='gray',
                                     annotation_text='Starting Value')
                
                fig_crisis.update_layout(
                    title = r['crisis'],
                    xaxis_title = 'Date', yaxis_title = 'Cumulative Return (1.0 = Starting)',
                    template = 'plotly_white', height = 350,
                    showlegend = False
                )
                st.plotly_chart(fig_crisis, use_container_width=True)

# ==========================================
# MODULE 5: BACKTESTER
# ==========================================

elif page == "📊 Backtester":
    st.header("📊 Portfolio Backtester")
    st.markdown("**Backtest a portfolio strategy against historical data and compare with a benchmark**")

    # --- Inputs ---
    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        bt_tickers_input = st.text_input("Tickers (Comma Separated)", value="VTI, BND, VNQ",
                                          key="bt_tickers")
        bt_tickers = [t.strip().upper() for t in bt_tickers_input.split(",") if t.strip()]

    with col_b2:
        st.write("")
        st.write("")
        run_bt = st.button("📊 Run Backtest", type="primary", use_container_width=True)

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        bt_start = st.date_input("Start Date", value=datetime(2015, 1, 1))
    with col_d2:
        bt_end = st.date_input("End Date", value=datetime.now())
    with col_d3:
        bt_rebal = st.selectbox("Rebalancing", ["none", "monthly", "quarterly", "annually"])
    with col_d4:
        bt_bench = st.text_input("Benchmark", value="SPY")

    # Weights
    if bt_tickers:
        st.markdown("**Portfolio Weights:**")
        wt_cols = st.columns(min(len(bt_tickers), 6))
        bt_weights = []
        for i, ticker in enumerate(bt_tickers):
            with wt_cols[i % len(wt_cols)]:
                w = st.number_input(f"{ticker} (%)", value=round(100.0/len(bt_tickers), 1),
                                    min_value=0.0, max_value=100.0, step=5.0, key=f"bw_{ticker}")
                bt_weights.append(w / 100.0)

    if run_bt and bt_tickers:
        bt_weights_arr = np.array(bt_weights)
        bt_weights_arr = bt_weights_arr / bt_weights_arr.sum()

        with st.spinner("🔄 Running backtest..."):
            bt_result = utils.run_backtest(
                bt_tickers, bt_weights_arr,
                bt_start.strftime('%Y-%m-%d'), bt_end.strftime('%Y-%m-%d'),
                bt_rebal, bt_bench.strip().upper()
            )

        if bt_result:
            st.session_state['bt_result'] = bt_result
            st.success("✅ **Backtest complete!**")
        else:
            st.error("❌ Backtest failed. Check your tickers and date range.")

    if 'bt_result' in st.session_state:
        bt = st.session_state['bt_result']

        # Key metrics
        st.divider()
        st.subheader("📊 Performance Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📈 Total Return", f"{bt['total_return']*100:.1f}%")
        m2.metric("📊 CAGR", f"{bt['cagr']*100:.2f}%")
        m3.metric("📉 Max Drawdown", f"{bt['max_drawdown']*100:.1f}%")
        m4.metric("⭐ Sharpe Ratio", f"{bt['sharpe']:.2f}")
        m5.metric("🎯 Win Rate", f"{bt['win_rate']*100:.1f}%")

        # Downside-focused risk-adjusted metrics
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("🛡️ Sortino Ratio", f"{bt.get('sortino', 0.0):.2f}",
                  help="Like Sharpe, but only penalizes downside volatility. Higher is better.")
        d2.metric("🏔️ Calmar Ratio", f"{bt.get('calmar', 0.0):.2f}",
                  help="CAGR divided by the absolute max drawdown. Return earned per unit of worst-case loss.")
        d3.metric("📉 Downside Dev.", f"{bt.get('downside_deviation', 0.0)*100:.1f}%",
                  help="Annualized standard deviation of negative returns only.")
        d4.metric("📊 Volatility", f"{bt.get('volatility', 0.0)*100:.1f}%",
                  help="Annualized standard deviation of all returns.")

        if 'benchmark_return' in bt:
            st.info(f"📊 **Benchmark ({bt['benchmark_name']}) Total Return:** {bt['benchmark_return']*100:.1f}%")

        # Cumulative return chart
        st.divider()
        st.subheader("📈 Cumulative Returns")
        fig_bt = go.Figure()

        fig_bt.add_trace(go.Scatter(
            x = bt['portfolio_cum'].index, y = bt['portfolio_cum'].values,
            mode = 'lines', name = 'Portfolio',
            line = dict(color='#636EFA', width=2)
        ))

        if 'benchmark_cum' in bt:
            fig_bt.add_trace(go.Scatter(
                x = bt['benchmark_cum'].index, y = bt['benchmark_cum'].values,
                mode = 'lines', name = f"Benchmark ({bt['benchmark_name']})",
                line = dict(color='#EF553B', width=2, dash='dash')
            ))

        fig_bt.update_layout(
            xaxis_title='Date', yaxis_title='Cumulative Return (1.0 = Start)',
            template='plotly_white', height=450, showlegend=True
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # Drawdown chart
        st.subheader("📉 Drawdown")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x = bt['drawdown_series'].index, y = bt['drawdown_series'].values * 100,
            mode = 'lines', fill = 'tozeroy', name = 'Drawdown',
            line = dict(color='#EF553B', width=1),
            fillcolor = 'rgba(239, 85, 59, 0.3)'
        ))
        fig_dd.update_layout(
            xaxis_title='Date', yaxis_title='Drawdown (%)',
            template='plotly_white', height=300
        )
        st.plotly_chart(fig_dd, use_container_width=True)

# ==========================================
# MODULE 6: RISK DASHBOARD
# ==========================================

elif page == "📉 Risk Dashboard":
    st.header("📉 Risk Dashboard")
    st.markdown("**Rolling risk metrics and CAPM factor decomposition**")

    # Inputs
    col_r1, col_r2, col_r3 = st.columns([3, 1, 1])
    with col_r1:
        rd_ticker_input = st.text_input("Ticker(s) (Comma Separated)", value="VTI",
                                         key="rd_tickers")
        rd_tickers = [t.strip().upper() for t in rd_ticker_input.split(",") if t.strip()]
    with col_r2:
        rd_window = st.number_input("Rolling Window (Days)", value=60, min_value=10, max_value=252, step=10)
    with col_r3:
        rd_bench = st.text_input("Benchmark", value="SPY", key="rd_bench")
    
    st.write("")
    run_rd = st.button("📉 Analyze Risk", type="primary")

    if run_rd and rd_tickers:
        with st.spinner("🔄 Computing rolling metrics..."):
            # Fetch data
            raw_rd = utils.get_stock_data(rd_tickers, period="3y")
            if raw_rd is not None:
                price_data_rd = utils.extract_price_data(raw_rd, prefer_adj_close=True)
                if price_data_rd is not None:
                    metrics = utils.compute_rolling_metrics(price_data_rd, window=rd_window,
                                                            benchmark_ticker=rd_bench.strip().upper())
                    if metrics:
                        st.session_state['rd_metrics'] = metrics
                        st.success("✅ **Rolling metrics computed!**")
                    
                    # Factor decomposition
                    bench_raw = utils.get_stock_data(rd_bench.strip().upper(), period="3y")
                    if bench_raw is not None:
                        bench_price = utils.extract_price_data(bench_raw, prefer_adj_close=True)
                        if bench_price is not None:
                            if isinstance(price_data_rd, pd.DataFrame):
                                port_ret = price_data_rd.pct_change().dropna().mean(axis=1)
                            else:
                                port_ret = price_data_rd.pct_change().dropna()
                            bench_ret = bench_price.iloc[:, 0].pct_change().dropna()
                            
                            factors = utils.compute_factor_decomposition(port_ret, bench_ret)
                            if factors:
                                st.session_state['rd_factors'] = factors
                                st.session_state['rd_bench_name'] = rd_bench.strip().upper()
                else:
                    st.error("❌ Could not extract price data.")
            else:
                st.error("❌ Could not fetch data. Check your tickers.")

    if 'rd_metrics' in st.session_state:
        metrics = st.session_state['rd_metrics']
        
        st.divider()
        st.subheader(f"📈 Rolling Metrics (Window: {rd_window} days)")

        # Rolling Volatility
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=metrics['rolling_vol'].index, y=metrics['rolling_vol'].values * 100,
            mode='lines', name='Rolling Volatility',
            line=dict(color='#636EFA', width=2)
        ))
        fig_vol.update_layout(
            title='Rolling Annualized Volatility',
            xaxis_title='Date', yaxis_title='Volatility (%)',
            template='plotly_white', height=350
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Rolling Sharpe
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(
            x=metrics['rolling_sharpe'].index, y=metrics['rolling_sharpe'].values,
            mode='lines', name='Rolling Sharpe',
            line=dict(color='#00CC96', width=2)
        ))
        fig_sharpe.add_hline(y=0, line_dash='dash', line_color='gray')
        fig_sharpe.add_hline(y=1.0, line_dash='dot', line_color='green',
                              annotation_text='Good (1.0)')
        fig_sharpe.update_layout(
            title='Rolling Annualized Sharpe Ratio',
            xaxis_title='Date', yaxis_title='Sharpe Ratio',
            template='plotly_white', height=350
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)

        # Rolling Sortino
        if 'rolling_sortino' in metrics and not metrics['rolling_sortino'].empty:
            fig_sortino = go.Figure()
            fig_sortino.add_trace(go.Scatter(
                x=metrics['rolling_sortino'].index, y=metrics['rolling_sortino'].values,
                mode='lines', name='Rolling Sortino',
                line=dict(color='#FFA15A', width=2)
            ))
            fig_sortino.add_hline(y=0, line_dash='dash', line_color='gray')
            fig_sortino.add_hline(y=1.0, line_dash='dot', line_color='green',
                                  annotation_text='Good (1.0)')
            fig_sortino.update_layout(
                title='Rolling Annualized Sortino Ratio (downside-only risk)',
                xaxis_title='Date', yaxis_title='Sortino Ratio',
                template='plotly_white', height=350
            )
            st.plotly_chart(fig_sortino, use_container_width=True)

        # Rolling Beta
        if 'rolling_beta' in metrics:
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(
                x=metrics['rolling_beta'].index, y=metrics['rolling_beta'].values,
                mode='lines', name='Rolling Beta',
                line=dict(color='#AB63FA', width=2)
            ))
            fig_beta.add_hline(y=1.0, line_dash='dash', line_color='gray',
                                annotation_text='Market Beta (1.0)')
            fig_beta.update_layout(
                title=f'Rolling Beta vs {metrics.get("benchmark_name", "SPY")}',
                xaxis_title='Date', yaxis_title='Beta',
                template='plotly_white', height=350
            )
            st.plotly_chart(fig_beta, use_container_width=True)

    # Factor Decomposition
    if 'rd_factors' in st.session_state:
        factors = st.session_state['rd_factors']
        bench_name = st.session_state.get('rd_bench_name', 'SPY')
        
        st.divider()
        st.subheader(f"🧬 CAPM Factor Decomposition (vs {bench_name})")
        
        f1, f2, f3, f4, f5 = st.columns(5)
        
        alpha_color = "🟢" if factors['alpha'] > 0 else "🔴"
        f1.metric(f"{alpha_color} Alpha (Annual)", f"{factors['alpha']*100:.2f}%",
                  help="Excess return not explained by market exposure. Positive = outperformance.")
        f2.metric("📊 Beta", f"{factors['beta']:.2f}",
                  help="Sensitivity to market movements. 1.0 = moves with market. >1 = more volatile.")
        f3.metric("📈 R²", f"{factors['r_squared']:.2f}",
                  help="How much of the portfolio's movement is explained by the benchmark. 1.0 = perfectly correlated.")
        f4.metric("📏 Tracking Error", f"{factors['tracking_error']*100:.1f}%",
                  help="Annualized standard deviation of the difference between portfolio and benchmark returns.")
        
        ir_color = "🟢" if factors['information_ratio'] > 0.5 else "🟡" if factors['information_ratio'] > 0 else "🔴"
        f5.metric(f"{ir_color} Info Ratio", f"{factors['information_ratio']:.2f}",
                  help="Active return per unit of tracking error. >0.5 is good, >1.0 is excellent.")
        
        with st.expander("📖 What do these metrics mean?"):
            st.markdown("""
            | Metric | Meaning |
            |---|---|
            | **Alpha** | The portfolio's excess return beyond what the market (beta) explains. Positive = you're generating value. |
            | **Beta** | Market sensitivity. β=1.0 means 1:1 with market. β=0.5 means half the market's movement. |
            | **R²** | % of portfolio variance explained by the benchmark. Higher = more correlated to market. |
            | **Tracking Error** | How much your returns deviate from the benchmark. Lower = closer tracking. |
            | **Information Ratio** | Active return per unit of risk taken vs benchmark. Higher = better risk-adjusted outperformance. |
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
        <p>2026 Jason Huang | Data Source: Yahoo Finance | Built with Streamlit & Python</p>
        </div>
        """, 
        unsafe_allow_html = True)
