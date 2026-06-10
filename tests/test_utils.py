
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
            'worst_case', 'cvar_95', 'prob_loss',
            'max_drawdown', 'sharpe_ratio', 'end_prices'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check DataFrame shape
        # Rows = time_horizon + 1 (Start day 0 + 20 days)
        # Cols = sim columns + 'Mean' 
        # Note: logic limits columns to min(simulations, MAX_LINES_TO_PLOT). 
        # utils.MAX_LINES_TO_PLOT is 50. 
        # If we ask for 50 simulations, we get 50 + 1 (Mean) = 51 columns.
        sim_df = result['simulation_df']
        assert len(sim_df) == time_horizon + 1
        assert 'Mean' in sim_df.columns
        
        # Check values are floats (sanity)
        assert isinstance(result['expected_price'], float)
        
        # Check new metrics
        assert isinstance(result['max_drawdown'], float)
        assert result['max_drawdown'] <= 0, "Max drawdown should be non-positive"
        assert isinstance(result['sharpe_ratio'], float)
        assert isinstance(result['end_prices'], np.ndarray)
        assert len(result['end_prices']) == simulations

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

class TestEfficientFrontier:
    def test_frontier_basic_shape(self):
        """Test that the efficient frontier returns correct structure."""
        np.random.seed(42)
        # 3 assets, annualized params
        mean_returns = np.array([0.10, 0.15, 0.20])
        # Create a valid covariance matrix
        data = np.random.normal(0, 0.01, (200, 3))
        cov_matrix = np.cov(data.T) * 252
        
        result = utils.compute_efficient_frontier(mean_returns, cov_matrix, num_points=20)
        
        assert result is not None
        assert 'frontier_vols' in result
        assert 'frontier_rets' in result
        assert len(result['frontier_vols']) >= 2
        assert len(result['frontier_rets']) >= 2
    
    def test_frontier_returns_none_for_equal_returns(self):
        """If all assets have the same expected return, frontier should return None."""
        mean_returns = np.array([0.10, 0.10, 0.10])
        cov_matrix = np.eye(3) * 0.04
        
        result = utils.compute_efficient_frontier(mean_returns, cov_matrix)
        assert result is None


class TestRiskParity:
    def test_risk_parity_basic(self):
        """Test risk parity returns valid structure with equal risk contributions."""
        np.random.seed(42)
        # Create synthetic returns data
        n_days = 252
        data = pd.DataFrame({
            'A': np.random.normal(0.0005, 0.01, n_days),
            'B': np.random.normal(0.0003, 0.02, n_days),
            'C': np.random.normal(0.0004, 0.015, n_days)
        })
        
        result = utils.optimize_portfolio_risk_parity(data, num_portfolios=100)
        
        assert result is not None
        assert 'opt_weights' in result
        assert 'risk_contributions' in result
        assert len(result['opt_weights']) == 3
        assert abs(result['opt_weights'].sum() - 1.0) < 0.01
        # Risk contributions should be roughly equal (within tolerance)
        rc = result['risk_contributions']
        assert np.std(rc) < 0.15, f"Risk contributions too unequal: {rc}"


class TestBlackLitterman:
    def test_bl_basic(self):
        """Test Black-Litterman returns valid structure."""
        np.random.seed(42)
        n_days = 252
        data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.015, n_days),
            'MSFT': np.random.normal(0.0008, 0.012, n_days),
            'GOOG': np.random.normal(0.0006, 0.018, n_days)
        })
        
        views = {'AAPL': 0.15}  # 15% annual return view on AAPL
        
        result = utils.optimize_portfolio_black_litterman(data, views, risk_free_rate=0.04, 
                                                           num_portfolios=100)
        
        assert result is not None
        assert 'opt_weights' in result
        assert 'adjusted_returns' in result
        assert 'equilibrium_returns' in result
        assert len(result['opt_weights']) == 3
        assert abs(result['opt_weights'].sum() - 1.0) < 0.01


class TestFactorDecomposition:
    def test_factor_decomposition_basic(self):
        """Test CAPM factor decomposition returns correct structure."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')
        market = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
        # Portfolio with beta ~1.2
        portfolio = market * 1.2 + np.random.normal(0.0001, 0.005, 252)
        portfolio = pd.Series(portfolio, index=dates)
        
        result = utils.compute_factor_decomposition(portfolio, market)
        
        assert result is not None
        assert 'alpha' in result
        assert 'beta' in result
        assert 'r_squared' in result
        assert 'tracking_error' in result
        assert 'information_ratio' in result
        # Beta should be approximately 1.2
        assert abs(result['beta'] - 1.2) < 0.3, f"Beta {result['beta']} too far from expected 1.2"
    
    def test_factor_insufficient_data(self):
        """Test returns None with insufficient data."""
        dates = pd.date_range('2020-01-01', periods=10, freq='B')
        p = pd.Series(np.random.normal(0, 0.01, 10), index=dates)
        b = pd.Series(np.random.normal(0, 0.01, 10), index=dates)
        
        result = utils.compute_factor_decomposition(p, b)
        assert result is None


class TestRollingMetrics:
    def test_rolling_metrics_basic(self):
        """Test rolling metrics returns correct structure."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 252))),
                            index=dates)
        
        result = utils.compute_rolling_metrics(prices, window=30, benchmark_ticker='SPY')
        
        assert result is not None
        assert 'rolling_vol' in result
        assert 'rolling_sharpe' in result
        assert len(result['rolling_vol']) > 0

    def test_rolling_metrics_dataframe_input(self):
        """Test rolling metrics works with DataFrame (multiple assets)."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')
        prices = pd.DataFrame({
            'A': 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 252))),
            'B': 50 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 252)))
        }, index=dates)
        
        result = utils.compute_rolling_metrics(prices, window=30, benchmark_ticker='SPY')
        
        assert result is not None
        assert len(result['rolling_vol']) > 0


# ==========================================
# Edge Case Tests
# ==========================================

class TestEdgeCases:
    def test_monte_carlo_single_simulation(self):
        """Monte Carlo should work with just 1 simulation."""
        np.random.seed(42)
        log_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        result = utils.run_monte_carlo_simulation(
            last_price=100.0, log_returns=log_returns,
            time_horizon=10, simulations=1
        )
        assert result is not None
        assert result['expected_price'] > 0
        assert len(result['end_prices']) == 1

    def test_optimize_portfolio_two_assets(self):
        """Portfolio optimizer should work with exactly 2 assets."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')
        prices = pd.DataFrame({
            'A': 100 + np.cumsum(np.random.randn(252) * 0.5),
            'B': 50 + np.cumsum(np.random.randn(252) * 0.3)
        }, index=dates)
        # Ensure prices stay positive
        prices = prices.clip(lower=1.0)
        
        result = utils.optimize_portfolio(prices, risk_free_rate=0.04, num_portfolios=50)
        assert result is not None
        assert abs(result['opt_weights'].sum() - 1.0) < 0.01

    def test_bl_no_matching_views(self):
        """Black-Litterman with views for non-existent tickers should use equilibrium."""
        np.random.seed(42)
        n_days = 252
        data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.015, n_days),
            'MSFT': np.random.normal(0.0008, 0.012, n_days)
        })
        
        views = {'NONEXISTENT': 0.20}  # No matching ticker
        result = utils.optimize_portfolio_black_litterman(data, views, risk_free_rate=0.04, 
                                                           num_portfolios=50)
        # Should still return a result using equilibrium returns
        assert result is not None
        assert abs(result['opt_weights'].sum() - 1.0) < 0.01

    def test_risk_parity_two_assets(self):
        """Risk parity should work with exactly 2 assets."""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.normal(0.0005, 0.01, 252),
            'B': np.random.normal(0.0003, 0.03, 252)  # much more volatile
        })
        
        result = utils.optimize_portfolio_risk_parity(data, num_portfolios=50)
        assert result is not None
        assert len(result['opt_weights']) == 2
        # Higher-vol asset should have lower weight
        assert result['opt_weights'][1] < result['opt_weights'][0], \
            f"Higher-vol asset B should have lower weight but got {result['opt_weights']}"

    def test_factor_decomposition_identical_series(self):
        """Factor decomposition of identical series should have beta=1, alpha=0."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
        
        result = utils.compute_factor_decomposition(returns, returns)
        assert result is not None
        assert abs(result['beta'] - 1.0) < 0.01, f"Beta should be ~1.0 but got {result['beta']}"
        assert abs(result['alpha']) < 0.01, f"Alpha should be ~0 but got {result['alpha']}"
        assert result['r_squared'] > 0.99, f"R² should be ~1.0 but got {result['r_squared']}"
        assert result['tracking_error'] < 0.01, f"Tracking error should be ~0 but got {result['tracking_error']}"

    def test_extract_price_data_with_none(self):
        """extract_price_data should handle None gracefully."""
        assert utils.extract_price_data(None) is None
    
    def test_empty_returns_risk_parity(self):
        """Risk parity with empty DataFrame should return None."""
        result = utils.optimize_portfolio_risk_parity(pd.DataFrame())
        assert result is None
    
    def test_empty_returns_bl(self):
        """Black-Litterman with empty DataFrame should return None."""
        result = utils.optimize_portfolio_black_litterman(pd.DataFrame(), {'A': 0.15}, 0.04)
        assert result is None


class TestDownsideMetrics:
    """Tests for the downside risk metric helpers."""

    def test_downside_deviation_no_losses_is_zero(self):
        """All-positive returns have zero downside deviation."""
        returns = np.array([0.01, 0.02, 0.015, 0.03])
        assert utils.downside_deviation(returns) == 0.0

    def test_downside_deviation_positive_with_losses(self):
        """Returns containing losses produce a positive downside deviation."""
        returns = np.array([0.02, -0.03, 0.01, -0.05, 0.04])
        assert utils.downside_deviation(returns) > 0.0

    def test_downside_deviation_empty(self):
        """Empty input returns 0.0 rather than raising."""
        assert utils.downside_deviation(np.array([])) == 0.0

    def test_downside_deviation_ignores_nan(self):
        """NaNs are dropped and do not break the computation."""
        clean = np.array([0.01, -0.02, 0.03])
        with_nan = np.array([0.01, np.nan, -0.02, 0.03])
        assert utils.downside_deviation(with_nan) == pytest.approx(
            utils.downside_deviation(clean))

    def test_sortino_positive_for_good_returns(self):
        """A series with positive mean and limited downside has a positive Sortino."""
        np.random.seed(1)
        returns = np.random.normal(0.001, 0.01, 252)
        assert utils.sortino_ratio(returns) > 0.0

    def test_sortino_zero_when_no_downside(self):
        """No downside deviation -> Sortino is defined as 0.0 (guarded)."""
        returns = np.array([0.01, 0.02, 0.03])
        assert utils.sortino_ratio(returns) == 0.0

    def test_calmar_basic(self):
        """Calmar = cagr / |max_drawdown|."""
        assert utils.calmar_ratio(0.20, -0.10) == pytest.approx(2.0)

    def test_calmar_zero_drawdown_guarded(self):
        """Zero drawdown returns 0.0 instead of dividing by zero."""
        assert utils.calmar_ratio(0.20, 0.0) == 0.0

    def test_omega_above_one_for_positive_skew(self):
        """More/larger gains than losses gives Omega > 1."""
        returns = np.array([0.05, 0.04, -0.01, 0.03, -0.02])
        assert utils.omega_ratio(returns) > 1.0

    def test_omega_infinite_when_no_losses(self):
        """No losses relative to threshold -> Omega is infinite."""
        returns = np.array([0.01, 0.02, 0.03])
        assert utils.omega_ratio(returns) == float('inf')


class TestMinCVaR:
    """Tests for the Minimum-CVaR optimizer."""

    def _sample_returns(self, n_days=300, seed=7):
        np.random.seed(seed)
        return pd.DataFrame({
            'LOWRISK': np.random.normal(0.0004, 0.006, n_days),
            'MIDRISK': np.random.normal(0.0006, 0.015, n_days),
            'HIGHRISK': np.random.normal(0.0008, 0.030, n_days),
        })

    def test_min_cvar_weights_valid(self):
        """Weights are non-negative and sum to 1."""
        result = utils.optimize_portfolio_min_cvar(
            self._sample_returns(), risk_free_rate=0.03, num_portfolios=50)
        assert result is not None
        w = result['opt_weights']
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-9).all()
        assert len(w) == 3

    def test_min_cvar_reports_tail_risk(self):
        """Result exposes a non-negative daily CVaR and the confidence level."""
        result = utils.optimize_portfolio_min_cvar(
            self._sample_returns(), risk_free_rate=0.03, num_portfolios=50)
        assert result is not None
        assert result['cvar'] >= 0.0
        assert result['alpha'] == 0.95

    def test_min_cvar_avoids_highest_tail_asset(self):
        """Tail-risk minimization should under-weight the most volatile asset."""
        result = utils.optimize_portfolio_min_cvar(
            self._sample_returns(), risk_free_rate=0.03, num_portfolios=50)
        assert result is not None
        w = result['opt_weights']
        # HIGHRISK is index 2; it should not be the dominant holding
        assert w[2] <= w[0] + 1e-6

    def test_min_cvar_empty_returns_none(self):
        """Empty input returns None."""
        assert utils.optimize_portfolio_min_cvar(pd.DataFrame(), 0.03) is None

    def test_min_cvar_single_asset_returns_none(self):
        """A single asset is insufficient for diversification; returns None."""
        single = pd.DataFrame({'ONLY': np.random.normal(0.0005, 0.01, 100)})
        assert utils.optimize_portfolio_min_cvar(single, 0.03) is None
