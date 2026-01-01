# Financial Analysis & Optimization Tool

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financialanalysistool-jasonhuang.streamlit.app)

A comprehensive, production-ready Streamlit application for financial analysis, portfolio optimization, and risk assessment. Built with modern quantitative finance techniques including Monte Carlo simulation and Modern Portfolio Theory.

## Overview

This application provides three powerful tools for investors and financial analysts:

1. **Stock Price Forecaster** - Predict future stock prices using Monte Carlo simulations
2. **Portfolio Optimizer** - Find optimal asset allocation using Modern Portfolio Theory
3. **Portfolio Rebalancer** - Calculate trades needed to rebalance your portfolio

All tools feature interactive visualizations, real-time data from Yahoo Finance, and comprehensive risk analysis.

## Key Features

### Stock Price Forecaster
- **Monte Carlo Simulation** using Geometric Brownian Motion (GBM)
- Configurable time horizons (5-365 trading days)
- Risk metrics including:
  - Value at Risk (VaR) at 95% confidence
  - Conditional Value at Risk (CVaR) / Expected Shortfall
  - Probability of loss calculations
- Interactive visualization of simulation paths
- Memory-optimized (stores only visualization subset)

### Portfolio Optimizer
- **Modern Portfolio Theory (MPT)** implementation
- Efficient Frontier construction
- Sharpe Ratio maximization using convex optimization
- Correlation analysis with heatmap visualization
- High correlation warnings for diversification
- Optimal asset allocation recommendations

### Portfolio Rebalancer
- Calculate required trades to reach target allocations
- Handles integer share constraints
- Real-time price fetching
- Cash balance management
- Portfolio value calculations

## Tech Stack

### Core Technologies
- **Python 3.8+** - Programming language
- **Streamlit** - Web application framework
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis
- **SciPy** - Scientific computing and optimization

### Visualization & Data
- **Plotly** - Interactive charts and graphs
- **yfinance** - Yahoo Finance API integration

### Key Libraries
- `scipy.optimize` - Portfolio optimization algorithms
- `numpy.random` - Monte Carlo simulation random number generation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Financial_Analysis_Tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run Stock_App.py
   ```

4. **Access the application**
   - The app will automatically open in your default web browser
   - Default URL: `http://localhost:8501`

## Usage

### Stock Price Forecaster

1. Select **"Stock Price Forecaster"** from the navigation
2. Enter a stock ticker symbol (e.g., AAPL, GOOGL, VOO)
3. Adjust simulation parameters in the sidebar:
   - Time Horizon: Number of trading days to forecast
   - Number of Simulations: More = more accurate but slower
   - Random Seed: For reproducible results
4. Click **"Start Simulation"**
5. Review the forecast chart and risk analysis metrics

### Portfolio Optimizer

1. Select **"Portfolio Optimizer"** from the navigation
2. Enter multiple tickers separated by commas (e.g., VTI, VEA, VNQ)
3. Configure optimization settings:
   - Number of Simulations: Portfolio combinations to test
   - Risk-Free Rate: Current risk-free rate (e.g., 3-5%)
4. Click **"Optimize"**
5. Review the Efficient Frontier chart and optimal allocation

### Portfolio Rebalancer

1. Select **"Portfolio Rebalancer"** from the navigation
2. Enter your current cash balance
3. Add your holdings:
   - Ticker symbol
   - Current number of shares
   - Target allocation percentage
4. Ensure target percentages sum to 100%
5. Click **"Calculate Rebalancing"**
6. Review the required trades and portfolio summary

## Project Structure

```
Financial_Analysis_Tool/
├── Stock_App.py              # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── DEPLOYMENT_CHECKLIST.md   # Deployment and maintenance guide
└── .gitignore               # Git ignore file (if present)
```

## Configuration

All configuration constants are defined at the top of `Stock_App.py` for easy customization:

- **API Settings**: Cache duration, data periods
- **Simulation Defaults**: Number of simulations, time horizons
- **Risk Parameters**: VaR confidence levels, correlation thresholds
- **Portfolio Settings**: Default cash balance, allocation tolerances

See the code comments for detailed explanations of each constant.

## Data Source

This application uses **Yahoo Finance** via the `yfinance` library for:
- Historical stock price data
- Company information
- Real-time price quotes

**Note**: Data availability depends on Yahoo Finance API. The application includes error handling for API failures.

## Disclaimer

This application is for **educational and informational purposes only**. The information presented does not constitute financial advice or recommendation to buy or sell any securities. All models are based on historical data and statistical assumptions, which do not guarantee future performance.

**Important**: Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## Credits & Acknowledgments

**Built with Python, Streamlit, and AI assistance from Gemini & Cursor**

### AI Assistance
This project was developed with the assistance of AI tools:
- **Google Gemini** - For code generation, optimization suggestions, and technical guidance
- **Cursor AI** - For pair programming, code completion, and debugging assistance

The AI tools were instrumental in:
- Code optimization and performance improvements
- Error handling and edge case management
- Documentation and code organization
- UX enhancements and user experience improvements

### Data & Libraries
- **Yahoo Finance** - Financial data provider
- **Streamlit** - Application framework
- **Plotly** - Visualization library
- **Open Source Community** - All Python libraries used

## License

This project is provided as-is for educational purposes. Please refer to individual library licenses for dependencies.

## Additional Resources

- **Deployment Guide**: See `DEPLOYMENT_CHECKLIST.md` for detailed deployment and maintenance instructions
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Yahoo Finance**: https://finance.yahoo.com/

## Troubleshooting

### Common Issues

**Issue**: "Error fetching stock data"
- **Solution**: Check your internet connection and verify the ticker symbol is correct
- **Note**: Yahoo Finance API may have rate limits

**Issue**: "No data found"
- **Solution**: Ensure ticker symbols are valid and try again
- **Tip**: Use popular tickers like AAPL, GOOGL, MSFT, VOO for testing

**Issue**: Application runs slowly
- **Solution**: Reduce the number of simulations in the sidebar settings
- **Tip**: 200-500 simulations provide a good balance of speed and accuracy

## Future Enhancements

Potential improvements for future versions:
- Additional risk metrics (Sortino Ratio, Maximum Drawdown)
- Support for cryptocurrency analysis
- Export functionality for results
- Historical backtesting capabilities
- Multi-currency support
- Custom date range selection
- Implementing Ledoit-Wolf Shrinkage or Black-Litterman to handle estimation errors and multicollinearity.

---

## About me

Jason Huang  
MSc in Data Science for Sustainability, NUS

