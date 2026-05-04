import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    @brief A standalone vectorized backtesting engine for quantitative strategies.
    
    This engine is designed to be strategy-agnostic. It takes position signals and 
    asset returns as input and computes standardized performance metrics.
    """
    
    @staticmethod
    def compute_performance(
        positions: Union[pd.Series, pd.DataFrame], 
        asset_returns: Union[pd.Series, pd.DataFrame], 
        transaction_cost_bps: float = 0.0,
        rebalance_threshold: float = 0.0,
        portfolio_size: int = 1
    ) -> Dict[str, Any]:
        """
        @brief Computes P&L and risk metrics with transaction costs and rebalancing thresholds.
        @param positions: Series or DataFrame of target positions/weights.
        @param asset_returns: Series or DataFrame of asset returns.
        @param transaction_cost_bps: Cost per unit of turnover in basis points.
        @param rebalance_threshold: Minimum change in weight to trigger a rebalance.
        @param portfolio_size: Divisor for capital allocation.
        """
        # Convert to DataFrame for unified handling if they are Series
        if isinstance(positions, pd.Series):
            positions = positions.to_frame()
        if isinstance(asset_returns, pd.Series):
            asset_returns = asset_returns.to_frame()

        # Handle Rebalancing Threshold (Path-dependent)
        if rebalance_threshold > 0:
            actual_positions = positions.copy()
            # We must iterate to respect path-dependency of 'current weight'
            # Note: For large DataFrames this might be slow, but it's necessary for correctness.
            curr_pos = np.zeros(positions.shape[1])
            for i in range(len(positions)):
                target_pos = positions.iloc[i].values
                # Check if change exceeds threshold
                if np.any(np.abs(target_pos - curr_pos) > rebalance_threshold):
                    curr_pos = target_pos
                actual_positions.iloc[i] = curr_pos
            positions = actual_positions

        # Calculate Gross Daily PnL
        # The P&L for T is positions[T-1] * returns[T].
        daily_asset_pnl = positions.shift(1).fillna(0) * asset_returns
        gross_pnl = daily_asset_pnl.sum(axis=1)

        # Calculate Transaction Costs
        # Turnover = | target_w_t - drifted_w_{t-1} |
        # drifted_w_{t-1} = w_{t-1} * exp(r_t)
        # Using exp(r_t) for log returns compatibility
        drifted_pos = positions.shift(1).fillna(0) * np.exp(asset_returns.fillna(0))
        turnover = (positions - drifted_pos).abs()
        
        # Apply cost (bps to decimal)
        c = transaction_cost_bps / 10000.0
        daily_costs = turnover.sum(axis=1) * c
        
        # Net Daily PnL
        pnl = gross_pnl - daily_costs
        
        # Scale by portfolio size
        scaled_pnl = pnl / portfolio_size
        
        # Aggregate stats
        ann_ret = scaled_pnl.mean() * 252
        ann_vol = scaled_pnl.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # Drawdown
        equity = np.exp(pnl.cumsum())
        running_max = equity.cummax()
        drawdown = (equity / running_max) - 1
        max_dd = drawdown.min()
        # Identify discrete trades: entry (pos flips from 0 to non-zero) to exit (pos flips to 0)
        is_active = (positions != 0).any(axis=1)
        starts = (is_active) & (is_active.shift(1).fillna(False) == False)
        ends = (is_active) & (is_active.shift(-1).fillna(False) == False)
        
        trade_returns = []
        hold_times = []
        
        start_indices = positions.index[starts]
        end_indices = positions.index[ends]
        
        # Match starts and ends to compute returns
        # This handles the case where a trade starts but doesn't end (open at end of sample)
        for s_idx in start_indices:
            # Find the first end index after or at this start
            future_ends = end_indices[end_indices >= s_idx]
            if not future_ends.empty:
                e_idx = future_ends[0]
                # Trade return = Cumulative daily returns across the window
                # We use the daily_pnl (unscaled) for the trade return
                window_pnl = pnl.loc[s_idx:e_idx]
                total_trade_ret = np.exp(window_pnl.sum()) - 1
                trade_returns.append(total_trade_ret)
                
                # Hold time (trading days)
                hold_times.append(len(window_pnl))

        trade_returns = np.array(trade_returns)
        avg_trade_ret = np.mean(trade_returns) if len(trade_returns) > 0 else 0
        trade_sharpe = (np.mean(trade_returns) / np.std(trade_returns)) if len(trade_returns) > 1 and np.std(trade_returns) > 0 else 0
        avg_hold_time = np.mean(hold_times) if len(hold_times) > 0 else 0
        
        return {
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(trade_returns),
            'trade_sharpe': trade_sharpe,
            'avg_trade_ret': avg_trade_ret,
            'avg_hold_time': avg_hold_time,
            'trade_returns': trade_returns.tolist(),
            'hold_times': hold_times,
            'daily_pnl': scaled_pnl,
            'equity_curve': equity
        }

    @staticmethod
    def aggregate_portfolio(daily_pnls: pd.DataFrame) -> Dict[str, Any]:
        """
        @brief Aggregates multiple daily P&L series into a single portfolio equity curve.
        """
        port_pnl = daily_pnls.sum(axis=1)
        
        ann_ret = port_pnl.mean() * 252
        ann_vol = port_pnl.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        return {
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'daily_pnl': port_pnl,
            'cum_return': port_pnl.cumsum()
        }
