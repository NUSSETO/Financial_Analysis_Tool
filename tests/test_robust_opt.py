
import pytest
import numpy as np
import pandas as pd
import sys
from unittest.mock import MagicMock

# Mock streamlit and yfinance as they are not needed for this logic
sys.modules['streamlit'] = MagicMock()
sys.modules['yfinance'] = MagicMock()

import utils

class TestRobustOptimization:
    def test_optimize_portfolio_robust_structure(self):
        """Test output keys and types."""
        # Create dummy returns
        dates = pd.date_range(start='2023-01-01', periods=100)
        returns = pd.DataFrame(np.random.normal(0, 0.01, (100, 3)), columns=['A', 'B', 'C'], index=dates)
        
        result = utils.optimize_portfolio_robust(returns, 0.03, 50)
        
        assert result is not None
        assert 'opt_weights' in result
        assert 'opt_ret' in result
        assert 'opt_std' in result
        assert 'results' in result # New key for simulation data
        
        assert isinstance(result['opt_weights'], np.ndarray)
        assert len(result['opt_weights']) == 3

    def test_optimize_portfolio_robust_constraints(self):
        """Test if weights sum to 1 and are non-negative."""
        dates = pd.date_range(start='2023-01-01', periods=50)
        # Create correlated returns
        data = np.random.normal(0, 0.02, (50, 4))
        returns = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'], index=dates)
        
        result = utils.optimize_portfolio_robust(returns, 0.03, 50)
        
        weights = result['opt_weights']
        
        # Check sum to 1 (allow small floating point error)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-4)
        
        # Check non-negative (allow small epsilon for numerical noise)
        assert np.all(weights >= -1e-5)

    def test_optimize_portfolio_robust_empty(self):
        """Test handling of empty input."""
        assert utils.optimize_portfolio_robust(None, 0.03, 50) is None
        assert utils.optimize_portfolio_robust(pd.DataFrame(), 0.03, 50) is None

    def test_optimization_logic_two_assets(self):
        """Test a simple 2-asset case where one is clearly better."""
        # Asset A: High return, low vol
        # Asset B: Low return, high vol
        # Result should heavily favor A or be 100% A if min variance dictates it.
        # Actually objective is Min Variance. so Asset A with low vol should be favored.
        
        dates = pd.date_range(start='2023-01-01', periods=100)
        
        # Asset A: sigma = 0.01
        ret_a = np.random.normal(0.001, 0.01, 100)
        # Asset B: sigma = 0.05
        ret_b = np.random.normal(0.0001, 0.05, 100)
        
        returns = pd.DataFrame({'A': ret_a, 'B': ret_b}, index=dates)
        
        result = utils.optimize_portfolio_robust(returns, 0.03, 50)
        weights = result['opt_weights']
        
        # A should have significantly higher weight than B
        assert weights[0] > weights[1]
