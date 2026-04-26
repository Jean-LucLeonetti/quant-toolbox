import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from typing import List, Tuple, Dict
from src.core.logger import setup_logger
from src.data.loader import load_universe_prices
from src.data.store import DataStore
from src.core.config import PairsConfig

logger = setup_logger(__name__)

import shutil
from datetime import datetime

class PairsExplorationPipeline:
    """
    @brief Pipeline to explore and validate pair trading candidates.
    """
    def __init__(self, universe_name: str = "sp500_utilities", pairs_config: PairsConfig = None):
        self.universe_name = universe_name
        self.pairs_config = pairs_config or PairsConfig()
        self.store = DataStore()
        self.run_dir = None

    def run(self, start: str = None, end: str = None, excluded_periods: List[Dict[str, str]] = None):
        """
        @brief Main entry point for the pairs exploration.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"output/runs/pairs_{self.universe_name}_{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Save a copy of the configuration
        config_path = "input/configuration.yaml"
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(self.run_dir, "configuration.yaml"))

        logger.info(f"Starting pairs exploration for universe: {self.universe_name}")
        logger.info(f"Outputs will be saved to: {self.run_dir}")
        
        # 1. Load Price Panel
        prices = load_universe_prices(self.universe_name, start, end)
        if prices is None or prices.empty:
            logger.error(f"Could not load price panel for {self.universe_name}")
            return False

        # Apply Excluded Periods
        if excluded_periods:
            for period in excluded_periods:
                p_start, p_end = period['start'], period['end']
                logger.info(f"Excluding period: {p_start} to {p_end}")
                prices = prices[~((prices.index >= p_start) & (prices.index <= p_end))]

        # 2. Step A: Add a returns panel
        logger.info("Computing log returns panel...")
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Save returns panel
        returns_path = os.path.join(self.store.processed_dir, f"{self.universe_name}_returns.parquet")
        returns.to_parquet(returns_path)
        logger.info(f"Saved returns panel to {returns_path}")

        # Compute and log annualized stats
        # Assuming 252 trading days
        ann_mean = returns.mean() * 252
        ann_vol = returns.std() * (252**0.5)
        
        # Correlation matrix
        corr_matrix = returns.corr()
        
        # Log stats (summary)
        logger.info("Return statistics computed.")
        report_dir = "output/reports"
        os.makedirs(report_dir, exist_ok=True)
        stats_path = os.path.join(report_dir, f"{self.universe_name}_pair_stats.csv")
        stats_df = pd.DataFrame({
            'Annualized_Mean_Return': ann_mean,
            'Annualized_Volatility': ann_vol
        })
        stats_df.to_csv(stats_path)
        
        # 3. Identify pairs with correlation > 0.8
        logger.info("Identifying candidate pairs with correlation > 0.8...")
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        candidates = upper.unstack().dropna()
        candidates = candidates[candidates > 0.8].sort_values(ascending=False)
        
        logger.info(f"Found {len(candidates)} candidate pairs.")

        # 4. Step B: Cointegration Testing & Half-Life
        logger.info(f"Performing {self.pairs_config.coint_mode} tests and computing mean reversion half-lives...")
        coint_results = []
        for (a, b), corr in candidates.items():
            # Cointegration Test
            score, pvalue, is_coint = self._perform_coint_test(prices[a], prices[b])
            
            # Step C: Compute Hedge Ratio and Spread
            # Regression: a = gamma * b + const
            y = prices[a]
            x = sm.add_constant(prices[b])
            model = sm.OLS(y, x).fit()
            gamma = model.params[b]
            spread = model.resid 
            
            # Step D: Compute Half-Life (Ornstein-Uhlenbeck fit)
            delta_s = spread.diff().dropna()
            s_lag = spread.shift(1).dropna()
            delta_s = delta_s.loc[s_lag.index]
            
            # Fit OU
            ou_model = sm.OLS(delta_s, sm.add_constant(s_lag)).fit()
            beta = ou_model.params.iloc[1]
            
            # Half-life = -ln(2) / beta
            half_life = -np.log(2) / beta if beta < 0 else np.nan
            
            coint_results.append({
                'Asset_1': a,
                'Asset_2': b,
                'Correlation': corr,
                'Hedge_Ratio': gamma,
                'Coint_T_Stat': score,
                'Coint_P_Value': pvalue,
                'Beta_OU': beta,
                'Half_Life_Days': half_life,
                'Is_Cointegrated': is_coint
            })
            
        coint_df = pd.DataFrame(coint_results)
        report_dir = os.path.join(self.run_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        coint_path = os.path.join(report_dir, f"{self.universe_name}_cointegration_results.csv")
        coint_df.to_csv(coint_path, index=False)
        logger.info(f"Enhanced cointegration results saved to {coint_path}")

        # 5. Step E: Eyeball top Robust pairs
        # We define robustness as p < 0.05 in both 2015-2020 and 2020-2025
        logger.info("Validating robustness across sub-periods (2015-2020 vs 2020-2025)...")
        robust_results = []
        prices_1 = prices.loc[:"2019-12-31"]
        prices_2 = prices.loc["2020-01-01":]

        for (a, b), corr in candidates.items():
            _, p1, sig1 = self._perform_coint_test(prices_1[a], prices_1[b])
            _, p2, sig2 = self._perform_coint_test(prices_2[a], prices_2[b])
            
            robust_results.append({
                'Asset_1': a,
                'Asset_2': b,
                'P_Val_2015_2020': p1,
                'P_Val_2020_2025': p2,
                'Is_Robust': sig1 and sig2
            })
            
        robust_df = pd.DataFrame(robust_results)
        coint_df = coint_df.merge(robust_df, on=['Asset_1', 'Asset_2'])
        
        # 6. Save Enhanced Results
        report_dir = os.path.join(self.run_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        coint_path = os.path.join(report_dir, f"{self.universe_name}_cointegration_results.csv")
        coint_df.to_csv(coint_path, index=False)
        
        # 7. Spread Plotting for Top Pairs
        # We plot the top 5 by full-period cointegration to allow visual inspection
        top_5_coint = coint_df.sort_values('Coint_P_Value').head(5)
        logger.info(f"Plotting spreads for top 5 cointegrated pairs...")
        self._plot_spread_series(prices, top_5_coint)

        # 8. Backtest Top Pairs and Store Metrics
        n = self.pairs_config.portfolio_size
        top_n = coint_df.sort_values('Coint_P_Value').head(n)
        logger.info(f"Performing naive backtests for top {n} pairs...")
        backtest_results, portfolio_returns = self._backtest_naive(prices, returns, top_n)
        
        # Merge backtest metrics into coint_df
        coint_df = coint_df.merge(backtest_results, on=['Asset_1', 'Asset_2'], how='left')

        # 9. Plot Portfolio Curve
        self._plot_portfolio_curve(portfolio_returns)

        # 10. Plot Normalized Prices & Z-Scores for top 5 (to avoid clutter)
        top_5 = top_n.head(5)
        self._plot_normalized_pairs(prices, list(zip(top_5['Asset_1'], top_5['Asset_2'])))
        self._plot_z_scores(prices, top_5, window=self.pairs_config.z_window)

        # 11. Generate Enhanced Markdown Ranking Report
        self._generate_ranking_report(coint_df)

        return True

    def _backtest_naive(self, prices: pd.DataFrame, returns: pd.DataFrame, pairs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        @brief Performs a naive backtest (no costs) and plots equity curves.
        @return (metrics_df, aggregated_daily_returns)
        """
        metrics = []
        all_daily_returns = pd.DataFrame(index=returns.index)
        output_dir = os.path.join(self.run_dir, "pairs", "backtests")
        os.makedirs(output_dir, exist_ok=True)
        
        for _, row in pairs_df.iterrows():
            a, b = row['Asset_1'], row['Asset_2']
            gamma = row['Hedge_Ratio']
            
            # 1. Spread and Hedge Ratio Generation
            if self.pairs_config.hedge_mode == "kalman_filter":
                spread, beta_series = self._compute_kalman_spread(prices[a], prices[b])
            else:
                y_ols = prices[a]
                x_ols = sm.add_constant(prices[b])
                model_ols = sm.OLS(y_ols, x_ols).fit()
                spread = model_ols.resid
                beta_series = pd.Series(row['Hedge_Ratio'], index=spread.index)
            
            # Z-Score Signal Generation
            w = self.pairs_config.z_window
            mu = spread.rolling(window=w).mean()
            sigma = spread.rolling(window=w).std()
            z = (spread - mu) / sigma
            
            # Position Rules
            pos = pd.Series(index=z.index, data=np.nan)
            pos[z < -self.pairs_config.z_entry] = 1
            pos[z > self.pairs_config.z_entry] = -1
            pos[z.abs() < self.pairs_config.z_exit] = 0
            pos = pos.ffill().fillna(0)
            
            # 2. Regime Filter (Sit out during high volatility)
            if self.pairs_config.regime_filter:
                # Use a 20-day rolling vol of the spread
                spread_vol = spread.rolling(window=20).std()
                vol_threshold = spread_vol.quantile(0.9) # Exclude top 10% vol days
                regime_mask = spread_vol < vol_threshold
                pos = pos * regime_mask.astype(int)
                logger.info(f"Regime filter active for {a}/{b}: sit-out threshold @ {vol_threshold:.4f}")
            
            # 3. Daily P&L Calculation
            pair_return = pos.shift(1) * (returns[a] - beta_series.shift(1) * returns[b])
            pair_return = pair_return.fillna(0)
            
            # Cumulative P&L
            cum_ret = pair_return.cumsum()
            equity_curve = np.exp(cum_ret) * 100
            
            # Stats
            total_return = (equity_curve.iloc[-1] / 100) - 1
            ann_ret = (1 + total_return)**(252/len(equity_curve)) - 1
            ann_vol = pair_return.std() * (255**0.5)
            sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0
            
            metrics.append({
                'Asset_1': a,
                'Asset_2': b,
                'Ann_Return': ann_ret,
                'Sharpe': sharpe
            })
            
            # Store daily returns for portfolio aggregation (scaled by 1/N)
            all_daily_returns[f"{a}_{b}"] = pair_return / len(pairs_df)
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            equity_curve.plot(ax=ax, color='green', lw=2)
            ax.set_title(f"Naive Equity Curve: {a} vs {b}", fontsize=14)
            ax.set_ylabel("Equity (Base 100)")
            ax.grid(True, alpha=0.3)
            ax.text(0.02, 0.95, f"Total Return: {total_return:.1%}\nAnn. Return: {ann_ret:.1%}\nSharpe: {sharpe:.2f}", 
                    transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            save_path = os.path.join(output_dir, f"{a}_{b}_equity.png")
            fig.savefig(save_path)
            plt.close(fig)
        portfolio_returns = all_daily_returns.sum(axis=1)
        return pd.DataFrame(metrics), portfolio_returns

    def _plot_z_scores(self, prices: pd.DataFrame, pairs_df: pd.DataFrame, window: int = 60):
        """
        @brief Plots the rolling z-score of the spread for selected pairs, overlaid with positions.
        """
        output_dir = os.path.join(self.run_dir, "pairs", "z_scores")
        os.makedirs(output_dir, exist_ok=True)
        
        for _, row in pairs_df.iterrows():
            a, b = row['Asset_1'], row['Asset_2']
            
            # Compute spread components
            if self.pairs_config.hedge_mode == "kalman_filter":
                spread, beta_series = self._compute_kalman_spread(prices[a], prices[b])
                logger.info(f"Using Kalman Filter for {a} vs {b}")
            else:
                y_ols = prices[a]
                x_ols = sm.add_constant(prices[b])
                model_ols = sm.OLS(y_ols, x_ols).fit()
                spread = model_ols.resid
            
            # Rolling Z-Score
            mu = spread.rolling(window=window).mean()
            sigma = spread.rolling(window=window).std()
            z = (spread - mu) / sigma
            
            # 2. Position Rules
            pos = pd.Series(index=z.index, data=np.nan)
            pos[z < -self.pairs_config.z_entry] = 1   # Long spread
            pos[z > self.pairs_config.z_entry] = -1   # Short spread
            pos[z.abs() < self.pairs_config.z_exit] = 0  # Exit
            pos = pos.ffill().fillna(0)
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot Z-Score
            ax1.plot(z.index, z, color='teal', alpha=0.6, label='Z-Score')
            ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax1.axhline(self.pairs_config.z_entry, color='red', linestyle='--', alpha=0.5)
            ax1.axhline(-self.pairs_config.z_entry, color='red', linestyle='--', alpha=0.5)
            ax1.axhline(self.pairs_config.z_exit, color='orange', linestyle=':', alpha=0.3)
            ax1.axhline(-self.pairs_config.z_exit, color='orange', linestyle=':', alpha=0.3)
            
            ax1.set_ylabel("Z-Score", color='teal')
            ax1.tick_params(axis='y', labelcolor='teal')
            ax1.set_ylim(-4, 4)
            
            # Plot Position Overlay
            ax2 = ax1.twinx()
            ax2.fill_between(pos.index, 0, pos, color='gray', alpha=0.2, label='Position')
            ax2.set_ylabel("Position (-1, 0, +1)", color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.set_ylim(-1.5, 1.5)
            
            plt.title(f"Z-Score & Position Rules (60d): {a} vs {b}", fontsize=14)
            fig.tight_layout()
            
            save_path = os.path.join(output_dir, f"{a}_{b}_zscore_pos.png")
            fig.savefig(save_path)
            plt.close(fig)

    def _compute_kalman_spread(self, y: pd.Series, x: pd.Series):
        """
        @brief Computes the dynamic hedge ratio and spread using a Kalman Filter.
        """
        # H_t = [x_t, 1]
        # State: [beta, alpha]
        obs_mat = np.vstack([x.values, np.ones(len(x))]).T
        
        # Hyperparameters (can be tuned)
        delta = 1e-5 
        Q = delta / (1 - delta) * np.eye(2) 
        R = 0.0001 # Small measurement noise
        
        # Initial states
        theta = np.zeros(2) 
        P = np.eye(2)
        
        betas = []
        spreads = []
        
        for t in range(len(y)):
            # Predict
            P = P + Q
            
            # Update
            H = obs_mat[t]
            y_obs = y.iloc[t]
            
            # Innovation
            e = y_obs - np.dot(H, theta)
            S = np.dot(H, np.dot(P, H.T)) + R
            K = np.dot(P, H.T) / S
            
            theta = theta + K * e
            P = P - np.outer(K, np.dot(H, P))
            
            betas.append(theta[0])
            spreads.append(e) 
            
            spreads.append(e) 
            
        return pd.Series(spreads, index=y.index), pd.Series(betas, index=y.index)

    def _plot_portfolio_curve(self, portfolio_returns: pd.Series):
        """
        @brief Plots the aggregated equity curve of the total portfolio.
        """
        cum_ret = (1 + portfolio_returns).cumprod()
        
        # Performance metrics
        ann_ret = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cum_ret.index, cum_ret, color='navy', linewidth=2, label=f'Portfolio (N={self.pairs_config.portfolio_size})')
        ax.axhline(1, color='black', alpha=0.3)
        ax.fill_between(cum_ret.index, 1, cum_ret, color='navy', alpha=0.1)
        
        ax.set_title(f"Aggregated Portfolio Equity Curve ({self.universe_name})\nAnn. Return: {ann_ret*100:.1f}% | Sharpe: {sharpe:.2f}")
        ax.set_ylabel("Growth of $1")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        report_dir = os.path.join(self.run_dir, "pairs", "backtests")
        os.makedirs(report_dir, exist_ok=True)
        fig.savefig(os.path.join(report_dir, f"{self.universe_name}_portfolio_equity.png"))
        plt.close(fig)
        logger.info(f"Portfolio equity curve saved to {report_dir}")

    def _perform_coint_test(self, y: pd.Series, x: pd.Series) -> Tuple[float, float, bool]:
        """
        @brief Universal cointegration test wrapper based on config.
        @return (statistic, p-value or proxy, is_significant)
        """
        if self.pairs_config.coint_mode == "johansen":
            stat, sig = self._perform_johansen_test(y, x)
            return stat, stat, sig # Return stat as proxy for p-value
        else:
            p_val, sig = self._perform_engle_granger(y, x)
            return p_val, p_val, sig

    def _perform_engle_granger(self, y: pd.Series, x: pd.Series) -> Tuple[float, bool]:
        """
        @brief Wrapper for Engle-Granger cointegration test.
        """
        _, p_val, _ = coint(y, x)
        return p_val, p_val < 0.05

    def _perform_johansen_test(self, y: pd.Series, x: pd.Series) -> Tuple[float, bool]:
        """
        @brief Performs Johansen cointegration test.
        @return (Trace Statistic, Is Significant at 5%)
        """
        # Johansen expects endog matrix
        df = pd.concat([y, x], axis=1).dropna()
        # det_order=0 (constant), k_ar_diff=1 (1 lag)
        result = coint_johansen(df, 0, 1)
        
        # We check the trace statistics for r=0 (null: no cointegration)
        trace_stat = result.lr1[0]
        crit_val_5pct = result.cvt[0, 1] # Index 1 is 95% confidence level (5% significance)
        
        # To keep it compatible with p-value reporting, we return the trace stat
        # but mark significance based on the critical value.
        return trace_stat, trace_stat > crit_val_5pct

    def _plot_spread_series(self, prices: pd.DataFrame, pairs_df: pd.DataFrame):
        """
        @brief Plots the spread series (Y - gamma*X) for selected pairs.
        """
        output_dir = os.path.join(self.run_dir, "pairs", "spreads")
        os.makedirs(output_dir, exist_ok=True)
        
        for _, row in pairs_df.iterrows():
            a, b = row['Asset_1'], row['Asset_2']
            gamma = row['Hedge_Ratio']
            
            # Compute spread (with constant from full sample refit)
            y = prices[a]
            x = sm.add_constant(prices[b])
            model = sm.OLS(y, x).fit()
            spread = model.resid
            
            fig, ax = plt.subplots(figsize=(12, 5))
            spread.plot(ax=ax, color='purple', alpha=0.8)
            ax.axhline(spread.mean(), color='black', linestyle='--')
            ax.set_title(f"Spread: {a} - {gamma:.3f} * {b}", fontsize=14)
            ax.set_ylabel("Spread Value")
            ax.grid(True, alpha=0.3)
            
            # Add some stats to plot
            ax.text(0.02, 0.95, f"p-val: {row['Coint_P_Value']:.4f}\nHalf-life: {row['Half_Life_Days']:.1f}d", 
                    transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            save_path = os.path.join(output_dir, f"{a}_{b}_spread.png")
            fig.savefig(save_path)
            plt.close(fig)

    def _generate_ranking_report(self, df: pd.DataFrame):
        """
        @brief Generates a readable markdown report of the ranked pairs.
        """
        report_path = os.path.join(self.run_dir, "reports", f"{self.universe_name}_pairs_ranking.md")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Sort by best overall p-value
        ranked_df = df.sort_values('Coint_P_Value').head(20)
        
        with open(report_path, "w") as f:
            f.write(f"# Pairs Robustness & Ranking Report: {self.universe_name}\n\n")
            f.write("Pairs are ranked by their full-period cointegration. **Robust** pairs are those with $p < 0.05$ in both 2015-2019 and 2020-2025.\n\n")
            
            # Format columns for display
            columns = [
                'Asset_1', 'Asset_2', 'Coint_P_Value', 'Half_Life_Days', 
                'Is_Robust', 'Ann_Return', 'Sharpe'
            ]
            # Ensure metrics exist in the ranking (only top 5 will have them initially)
            for col in ['Ann_Return', 'Sharpe']:
                if col not in ranked_df.columns:
                    ranked_df[col] = np.nan

            display_df = ranked_df[columns].copy()
            
            # Style
            def style_robust(row):
                return f"**{row['Is_Robust']}**" if row['Is_Robust'] else f"{row['Is_Robust']}"
            
            display_df['Is_Robust'] = display_df.apply(style_robust, axis=1)
            display_df['Coint_P_Value'] = display_df['Coint_P_Value'].map(lambda x: f"**{x:.4f}**" if x < 0.05 else f"{x:.4f}")
            display_df['Half_Life_Days'] = display_df['Half_Life_Days'].map(lambda x: f"{x:.1f}" if not np.isnan(x) else "∞")
            display_df['Ann_Return'] = display_df['Ann_Return'].map(lambda x: f"{x:.1%}" if not np.isnan(x) else "-")
            display_df['Sharpe'] = display_df['Sharpe'].map(lambda x: f"{x:.2f}" if not np.isnan(x) else "-")
            
            f.write(display_df.to_markdown(index=False))
            f.write("\n\n## Plots\n")
            f.write("Visual verification plots can be found in:\n")
            f.write("- **Spread Time Series**: `pairs/spreads/`\n")
            f.write("- **Normalized Prices**: `pairs/normalized/`\n")
            f.write("- **Rolling Z-Scores**: `pairs/z_scores/`\n")
            f.write("- **Naive Backtests**: `pairs/backtests/`\n")

        logger.info(f"Ranking report saved to {report_path}")

    def _plot_normalized_pairs(self, prices: pd.DataFrame, pairs: List[Tuple[str, str]]):
        """
        @brief Generates normalized price plots for the selected pairs.
        """
        output_dir = os.path.join(self.run_dir, "pairs", "normalized")
        os.makedirs(output_dir, exist_ok=True)
        
        for a, b in pairs:
            logger.info(f"Plotting normalized price for {a} vs {b}")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Normalize to 100
            (prices[a] / prices[a].iloc[0] * 100).plot(ax=ax, label=a, color='blue', alpha=0.8)
            (prices[b] / prices[b].iloc[0] * 100).plot(ax=ax, label=b, color='red', alpha=0.8)
            
            ax.set_title(f"{a} vs {b} - Normalized Price (Start=100)", fontsize=14)
            ax.set_ylabel("Normalized Price")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            save_path = os.path.join(output_dir, f"{a}_{b}_normalized.png")
            fig.savefig(save_path)
            plt.close(fig)
            
        logger.info(f"Normalized plots saved to {output_dir}")
