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
        cum_ret = pnl.cumsum() # Use unscaled pnl for individual drawdown
        equity = np.exp(cum_ret)
        running_max = equity.cummax()
        drawdown = (equity / running_max) - 1
        max_dd = drawdown.min()
        
        # Trades
        entries = ((positions != 0) & (positions.shift(1) == 0)).sum()
        
        return {
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_trades': entries,
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
