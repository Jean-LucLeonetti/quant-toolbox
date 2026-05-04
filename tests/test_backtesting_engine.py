import pytest
import pandas as pd
import numpy as np
from src.backtesting.engine import BacktestEngine

@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-01", periods=10)
    returns = pd.DataFrame(0.0, index=dates, columns=['A', 'B'])
    positions = pd.DataFrame(0.0, index=dates, columns=['A', 'B'])
    return dates, returns, positions

def test_transaction_costs_simple(sample_data):
    dates, returns, positions = sample_data
    # Open positions on day 1
    positions.loc[dates[0], :] = [1.0, -1.0]
    # Stay on day 2
    positions.loc[dates[1], :] = [1.0, -1.0]
    # Close on day 3
    positions.loc[dates[2], :] = [0.0, 0.0]
    
    # 10 bps = 0.001
    perf = BacktestEngine.compute_performance(
        positions=positions,
        asset_returns=returns,
        transaction_cost_bps=10.0
    )
    
    # Turnover:
    # T1: 0 -> 1, 0 -> -1 | sum = 2. Cost = 0.002
    # T2: 1 -> 1, -1 -> -1 | sum = 0. Cost = 0
    # T3: 1 -> 0, -1 -> 0 | sum = 2. Cost = 0.002
    # Total = 0.004
    assert np.isclose(perf['daily_pnl'].sum(), -0.004)

def test_rebalance_threshold(sample_data):
    dates, returns, positions = sample_data
    # 0.05 threshold
    # Day 1: 1.0 (Trade)
    # Day 2: 1.02 (Skip)
    # Day 3: 1.06 (Trade)
    positions['A'] = [1.0, 1.02, 1.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    perf = BacktestEngine.compute_performance(
        positions=positions,
        asset_returns=returns,
        rebalance_threshold=0.05,
        transaction_cost_bps=100.0 # 1% = 0.01
    )
    
    # Expected trades:
    # T1: 0 -> 1.0. Cost = 0.01
    # T2: 1.02 vs current 1.0 (diff 0.02 < 0.05). Actual pos stays 1.0. Cost = 0
    # T3: 1.06 vs current 1.0 (diff 0.06 > 0.05). Actual pos becomes 1.06. Cost = |1.06 - 1.0| * 0.01 = 0.0006
    # T4: 0.0 vs current 1.06 (diff 1.06 > 0.05). Actual pos becomes 0.0. Cost = 1.06 * 0.01 = 0.0106
    # Total = 0.01 + 0.0006 + 0.0106 = 0.0212
    assert np.isclose(perf['daily_pnl'].sum(), -0.0212)

def test_drift_calculation():
    dates = pd.date_range("2023-01-01", periods=2)
    # Asset returns 10% on day 2
    returns = pd.DataFrame({'A': [0.0, np.log(1.1)]}, index=dates)
    # Hold 1.0 units on day 1
    # Target 1.1 units on day 2
    positions = pd.DataFrame({'A': [1.0, 1.1]}, index=dates)
    
    perf = BacktestEngine.compute_performance(
        positions=positions,
        asset_returns=returns,
        transaction_cost_bps=100.0 # 0.01
    )
    
    # Day 1: 0 -> 1.0. Cost = 0.01. PnL = -0.01
    # Day 2: 
    #   Starting weight: 1.0
    #   Drifted weight: 1.0 * exp(log(1.1)) = 1.1
    #   Target weight: 1.1
    #   Turnover: |1.1 - 1.1| = 0.0
    #   Gross PnL: 1.0 * log(1.1) = 0.09531...
    #   Net PnL: 0.09531 - 0 = 0.09531
    # Total PnL: 0.09531 - 0.01 = 0.08531
    expected = np.log(1.1) - 0.01
    assert np.isclose(perf['daily_pnl'].sum(), expected)

def test_portfolio_aggregation():
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({
        'pair1': [0.01, 0.02, -0.01, 0.03, 0.0],
        'pair2': [-0.01, 0.0, 0.02, 0.01, 0.05]
    }, index=dates)
    
    perf = BacktestEngine.aggregate_portfolio(df)
    assert len(perf['daily_pnl']) == 5
    assert np.isclose(perf['daily_pnl'].iloc[0], 0.0) # 0.01 - 0.01
    assert perf['sharpe'] > 0
