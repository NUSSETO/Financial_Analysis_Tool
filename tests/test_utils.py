
import sys
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
import pytest

# --- Mocking dependencies before importing utils ---
# This ensures that when utils is imported, it uses these mocks instead of trying to import the actual packages
# which might not be installed in the test environment.

# Mock Streamlit
mock_st = MagicMock()
# Mock cache_data decorator to just return the function as is
def mock_cache(ttl=None, **kwargs):
    return lambda func: func
mock_st.cache_data = mock_cache
sys.modules['streamlit'] = mock_st

# Mock YFinance
sys.modules['yfinance'] = MagicMock()

# Now it is safe to import utils
import utils

# --- Test Cases ---

class TestExtractPriceData:
    def test_extract_price_data_none(self):
        """Test with None input."""
        assert utils.extract_price_data(None) is None

    def test_extract_price_data_empty(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        assert utils.extract_price_data(df) is None

    def test_extract_price_data_valid_multiindex(self):
        """Test with simulated multi-index DataFrame from yfinance (batch download)."""
        # Structure: Columns level 0 = Attributes (Adj Close, Close), Level 1 = Tickers
        arrays = [
            ["Adj Close", "Adj Close", "Close", "Close"],
            ["AAPL", "GOOG", "AAPL", "GOOG"]
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=["Attribute", "Ticker"])
        
        data = pd.DataFrame(
            np.array([[150, 2800, 149, 2790], [152, 2820, 151, 2810]]), 
            columns=index
        )
        
        result = utils.extract_price_data(data, prefer_adj_close=True)
        
        assert result is not None
        assert result.shape == (2, 2)
        assert list(result.columns) == ["AAPL", "GOOG"]
        # Check values match Adj Close
        assert result.iloc[0, 0] == 150 # AAPL
        assert result.iloc[0, 1] == 2800 # GOOG

    def test_extract_price_data_single_ticker(self):
        """Test with single ticker dataframe (flat columns)."""
        data = pd.DataFrame({
            "Adj Close": [100, 101],
            "Close": [99, 100],
            "Volume": [1000, 1200]
        })
        # Simulate yf.download for single ticker often just returns flat cols
        result = utils.extract_price_data(data, prefer_adj_close=True)
        
        assert result is not None
        assert "Adj Close" in result.columns

class TestMonteCarloSimulation:
    def test_monte_carlo_output_structure(self):
        """Test if the simulation returns the correct dictionary structure."""
        # Setup inputs
        last_price = 100.0
        time_horizon = 20
        simulations = 50
        # Dummy log returns
        log_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        result = utils.run_monte_carlo_simulation(last_price, log_returns, time_horizon, simulations)
        
        # Check Keys
        expected_keys = [
            'simulation_df', 'expected_price', 'median_price', 
            'worst_case', 'cvar_95', 'prob_loss'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check DataFrame shape
        # Rows = time_horizon + 1 (Start day 0 + 20 days)
        # Cols = sim columns + 'Mean' 
        # Note: logic limits columns to min(simulations, MAX_lines_TO_PLOT). 
        # utils.MAX_lines_TO_PLOT is 50. 
        # If we ask for 50 simulations, we get 50 + 1 (Mean) = 51 columns.
        sim_df = result['simulation_df']
        assert len(sim_df) == time_horizon + 1
        assert 'Mean' in sim_df.columns
        
        # Check values are floats (sanity)
        assert isinstance(result['expected_price'], float)

class TestRebalancingPlan:
    def test_calculate_rebalancing_basics(self):
        """Test the core rebalancing math."""
        current_cash = 10000.0
        
        valid_rows = pd.DataFrame({
            "Ticker": ["VTI", "BND"],
            "Shares": [10, 20],
            "Target (%)": [60.0, 40.0]
        })
        
        current_prices = {
            "VTI": 200.0,
            "BND": 100.0
        }
        
        # Calculation:
        # Equity = 10000 (cash) + 10*200 (VTI) + 20*100 (BND) 
        #        = 10000 + 2000 + 2000 = 14000
        
        # Target VTI: 60% of 14000 = 8400
        # Target BND: 40% of 14000 = 5600
        
        # New Shares VTI: floor(8400 / 200) = 42
        # New Shares BND: floor(5600 / 100) = 56
        
        # Trades:
        # VTI: 42 - 10 = +32
        # BND: 56 - 20 = +36
        
        result = utils.calculate_rebalancing_plan(current_cash, valid_rows, current_prices)
        
        assert 'error' not in result
        
        df = result['results_df']
        assert not df.empty
        
        # Check VTI
        vti_row = df[df['Ticker'] == "VTI"].iloc[0]
        assert vti_row['New Shares'] == 42
        assert vti_row['Trade (+/-)'] == 32
        
        # Check BND
        bnd_row = df[df['Ticker'] == "BND"].iloc[0]
        assert bnd_row['New Shares'] == 56
        assert bnd_row['Trade (+/-)'] == 36
        
        # Check Projected Cash
        # Cost VTI: 42 * 200 = 8400
        # Cost BND: 56 * 100 = 5600
        # Total Invested: 14000
        # Projected Cash: 14000 (Total Equity) - 14000 (Invested) = 0
        
        assert result['projected_cash'] == 0.0

    def test_rebalancing_negative_equity_check(self):
        """Test failure case with zero equity."""
        current_cash = 0
        valid_rows = pd.DataFrame({
             "Ticker": ["A"],
             "Shares": [0],
             "Target (%)": [100.0]
        })
        prices = {"A": 100.0}
        
        result = utils.calculate_rebalancing_plan(current_cash, valid_rows, prices)
        
        assert 'error' in result
