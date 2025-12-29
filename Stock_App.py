from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go 
import requests
import scipy.optimize as sco 
import streamlit as st

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
                ["Stock Price Forecaster", "Portfolio Optimizer"], 
                horizontal = True,
                label_visibility = "collapsed")

st.markdown("---")

# ==========================================
# FMP DATA LOADER 
# ==========================================

@st.cache_data(ttl = 3600)

def get_stock_data(tickers, period):
    
    # --- 1. API Key Validation ---
    api_key = st.secrets.get("FMP_API_KEY")
    
    if not api_key:
        st.error("Missing 'FMP_API_KEY' in .streamlit/secrets.toml")
        st.stop()

    # --- 2. Date Calculation ---
    end_date = datetime.now()
    if period.endswith("y"):
        try:
            years = int(period[:-1])
            days = years * 365
            
        except ValueError:
            days = 30
    else:
        days = 30 # Default to 30 days

    start_date = end_date - timedelta(days = days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # --- 3. Input Handling ---
    # Determine if request is single (str) or multiple (list)
    if isinstance(tickers, str):
        ticker_list = [tickers]
        is_single_request = True
    else:
        ticker_list = tickers
        is_single_request = False

    combined_data = {}

    # --- 4. Fetch Loop ---
    for t in ticker_list:
        url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/{t}"
               f"?from={start_str}&to={end_str}&apikey={api_key}")
        
        try:
            response = requests.get(url, timeout = 5)
            if response.status_code != 200:
                print(f"FMP Error for {t}: {response.status_code}")
                continue
            
            data_json = response.json()
            if "historical" not in data_json:
                continue

            # Parse JSON
            df = pd.DataFrame(data_json["historical"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            
            # Store Adjusted Close
            combined_data[t] = df["adjClose"]

        except Exception as e:
            print(f"Error fetching {t}: {e}")
            continue

    # --- 5. Return Formatting (Critical Step) ---
    if not combined_data:
        st.error(f"No data found for tickers: {tickers}")
        st.stop()
        return None

    result_df = pd.DataFrame(combined_data)

    # STRICT RETURN TYPE:
    # If user input was "AAPL" (str) -> Return Series (crucial for Forecaster math)
    if is_single_request:
        if tickers in result_df.columns:
            return result_df[tickers]
        return result_df.iloc[:, 0] # Fallback
    
    # If user input was ["AAPL", "GOOG"] (list) -> Return DataFrame
    else:
        return result_df

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
            closing_prices = get_stock_data(ticker, period = "1y")

            if closing_prices is None or closing_prices.empty:
                st.error(f"Ticker '{ticker}' not found or API unavailable.")
                
            else:

                # Calculate Log Returns for Geometric Brownian Motion parameters
                log_returns = np.log(closing_prices / closing_prices.shift(1)).dropna()
                mu = log_returns.mean()
                sigma = log_returns.std()
                last_price = float(closing_prices.iloc[-1]) 
                
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
    
                # --- Output Visualization ---
                st.success(f"Simulation complete! Reference Price: ${last_price:.2f}")
                st.subheader(f"Projected Paths for {ticker}")

                # Initiate the figure
                fig = go.Figure()
                
                # Performance Optimization: Limit rendered traces to prevent frontend lag
                max_lines_to_plot = 50
                columns_to_plot = list(simulation_df.columns[:min(simulations, max_lines_to_plot)])
                
                # Make sure the worst scenario is in the plot
                worst_col = (end_prices - worst_case).abs().idxmin()
                
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
                fig.update_layout(title = f"{simulations} Monte Carlo Simulations Scenarios",
                                  xaxis_title = "Trading Days into Future", 
                                  yaxis_title = "Price (USD)",
                                  xaxis = dict(range = [0, time_horizon]),  
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
# MODULE 2: PORTFOLIO OPTIMIZER
# ==========================================

elif page == "Portfolio Optimizer":
    st.header("Modern Portfolio Theory (Efficient Frontier)")
    
    # --- Sidebar Settings ---  
    st.sidebar.markdown("---")
    st.sidebar.header("Optimization Settings")
    num_portfolios = st.sidebar.slider("Number of Simulations", 
                                       1000, 10000, 5000,
                                       help = "Higher simulations produce a more accurate Efficient Frontier.")

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
            # Fetch data for 3 years to calculate correlation matrix
            np.random.seed(int(round(seed)))
            data = get_stock_data(tickers, period = "3y")

            if data is None or not isinstance(data, pd.DataFrame) or data.shape[1] < 2:
                st.error(f"Not enough valid data. Please check your tickers or try again later (API may be unavailable)")
                st.stop()

            # --- MPT Calculations ---
            MPT_returns = data.pct_change().dropna(how = 'any')
            mean_returns = MPT_returns.mean() * 252 # Annualized
            cov_matrix = MPT_returns.cov() * 252    # Annualized

            # Objective Functions for Scipy Optimize
            def portfolio_performance(weights, mean_returns, cov_matrix):
                returns = np.sum(mean_returns * weights)
                std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return returns, std

            def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate = 0.02):
                p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)         
                return - (p_ret - risk_free_rate) / p_std

            # Sequential Least Squares Programming Optimization for Max Sharpe Ratio
            num_assets = len(data.columns)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) 
            bounds = tuple((0, 1) for _ in range(num_assets)) 
            init_guess = num_assets * [1. / num_assets]

            opt_results = sco.minimize(neg_sharpe, init_guess, 
                                       args = (mean_returns, cov_matrix), 
                                       method = 'SLSQP', 
                                       bounds = bounds, 
                                       constraints = constraints)
            
            if not opt_results.success:
                st.warning(f"Optimization failed: {opt_results.message}")
                st.stop()

            opt_weights = opt_results.x
            opt_ret, opt_std = portfolio_performance(opt_weights, mean_returns, cov_matrix)
                    
            # --- Vectorized Portfolio Simulation ---
            # 1. Generate random weight matrix (num_portfolios x num_assets)
            weights = np.random.random((num_portfolios, num_assets))
            weights /= np.sum(weights, axis=1)[:, np.newaxis] # Normalize to sum=1

            # 2. Compute Returns (Dot Product)
            sim_returns = np.dot(weights, mean_returns)

            # 3. Compute Volatility using Einstein Summation (einsum)
            # Efficient calculation of diagonal elements of (w @ Sigma @ w.T)
            sim_variances = np.einsum('ij,jk,ik->i', weights, cov_matrix, weights)
            sim_stds = np.sqrt(sim_variances)

            # 4. Compute Sharpe Ratios
            risk_free_rate = 0.02
            sim_sharpes = (sim_returns - risk_free_rate) / sim_stds

            # Stack results: [Returns, Volatility, Sharpe]
            results = np.vstack([sim_returns, sim_stds, sim_sharpes])

            # --- Visualization ---
            st.subheader("Efficient Frontier")
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
                                     name = 'Max Sharpe (Optimal)'))

            fig.update_layout(title = "Risk vs. Return Analysis",
                              xaxis_title = "Volatility (Annualized Std Dev)", 
                              yaxis_title = "Expected Annual Return",          
                              height = 600,
                              legend = dict(yanchor = "top", 
                                            y = 0.99,
                                            xanchor = "left", 
                                            x = 0.01,
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
                        We simulate thousands of random combinations to find the "Efficient Frontier" — the curve where you get the **maximum possible return** for a given level of risk.
                        
                        **The Goal:**
                        Maximize the **Sharpe Ratio**:
                        $$
                        \text{Sharpe} = \frac{R_p - R_f}{\sigma_p}
                        $$
                        - $R_p$: Portfolio Return
                        - $R_f$: Risk-Free Rate (Fixed at 2% for this model)
                        - $\sigma_p$: Portfolio Risk (Volatility)
                        """)

            # --- Final Allocation Output ---
            st.subheader("Optimal Asset Allocation")
                    
            allocation_df = pd.DataFrame({"Ticker": data.columns, 
                                          "Weight": opt_weights})

            allocation_df = allocation_df.sort_values(by = "Weight", 
                                                      ascending = False)

            allocation_df['Weight'] = allocation_df['Weight'].apply(lambda x: f"{x*100:.2f}%")

            st.table(allocation_df.set_index('Ticker'))

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
        <p>2025 Jason Huang | Data Source: Financial Modeling Prep | Built with Streamlit, Python & Gemini</p>
        </div>
        """, 
        unsafe_allow_html = True)