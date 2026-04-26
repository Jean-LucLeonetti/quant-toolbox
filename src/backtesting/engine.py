import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
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
        positions: pd.Series, 
        asset_returns: pd.Series, 
        portfolio_size: int = 1
    ) -> Dict[str, Any]:
        """
        @brief Computes P&L and risk metrics for a single position series.
        @param positions: Series of target positions (+1, 0, -1)
        @param asset_returns: Series of daily asset returns
        @param portfolio_size: Divisor for capital allocation (if part of a multi-asset portfolio)
        @return Dictionary of metrics including ann_return, ann_vol, sharpe, and daily_pnl
        """
        # Ensure we shift positions to avoid look-ahead bias (trade occurs at T+1)
        # Note: In-sample we might already have shifted, but the engine enforces it for safety.
        # However, to be flexible, we assume 'positions' are the held state at the close of T.
        # The P&L for T+1 is positions[T] * returns[T+1].
        pnl = positions.shift(1).fillna(0) * asset_returns
        
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
        is_active = positions != 0
        starts = (is_active) & (positions.shift(1).fillna(0) == 0)
        ends = (is_active) & (positions.shift(-1).fillna(0) == 0)
        
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
