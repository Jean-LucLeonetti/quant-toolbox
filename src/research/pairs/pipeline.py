import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from typing import List, Tuple, Dict
from src.core.logger import setup_logger
from src.data.loader import load_universe_prices
from src.data.store import DataStore

logger = setup_logger(__name__)

class PairsExplorationPipeline:
    """
    @brief Pipeline to explore and validate pair trading candidates.
    """
    def __init__(self, universe_name: str = "sp500_utilities"):
        self.universe_name = universe_name
        self.store = DataStore()

    def run(self, start: str = None, end: str = None):
        """
        @brief Main entry point for the pairs exploration.
        """
        logger.info(f"Starting pairs exploration for universe: {self.universe_name}")
        
        # 1. Load Price Panel
        prices = load_universe_prices(self.universe_name, start, end)
        if prices is None or prices.empty:
            logger.error(f"Could not load price panel for {self.universe_name}")
            return False

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

        # 4. Step B: Cointegration Testing (Engle-Granger) & Half-Life
        logger.info("Performing Engle-Granger tests and computing mean reversion half-lives...")
        coint_results = []
        for (a, b), corr in candidates.items():
            # Engle-Granger Cointegration Test
            score, pvalue, _ = coint(prices[a], prices[b])
            
            # Step C: Compute Hedge Ratio and Spread
            # Regression: a = gamma * b + const
            y = prices[a]
            x = sm.add_constant(prices[b])
            model = sm.OLS(y, x).fit()
            gamma = model.params[b]
            spread = model.resid # a - gamma*b - const
            
            # Step D: Compute Half-Life (Ornstein-Uhlenbeck fit)
            # Regression: delta_S = alpha + beta * S_{t-1} + e
            delta_s = spread.diff().dropna()
            s_lag = spread.shift(1).dropna()
            # Ensure indices match
            delta_s = delta_s.loc[s_lag.index]
            
            # Fit OU
            ou_model = sm.OLS(delta_s, sm.add_constant(s_lag)).fit()
            beta = ou_model.params[0] # The coefficient on spread.shift(1)... wait, sm.add_constant adds const at index 0 normally?
            # Actually, sm.add_constant adds it at the start. 
            # delta_s = alpha (index 0) + beta (index 1) * s_lag
            beta = ou_model.params.iloc[1]
            
            # Half-life = -ln(2) / beta
            # beta should be negative for mean reversion
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
                'Is_Cointegrated': pvalue < 0.05
            })
            
        coint_df = pd.DataFrame(coint_results)
        report_dir = "output/reports"
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
            _, p1, _ = coint(prices_1[a], prices_1[b])
            _, p2, _ = coint(prices_2[a], prices_2[b])
            
            robust_results.append({
                'Asset_1': a,
                'Asset_2': b,
                'P_Val_2015_2020': p1,
                'P_Val_2020_2025': p2,
                'Is_Robust': (p1 < 0.05) and (p2 < 0.05)
            })
            
        robust_df = pd.DataFrame(robust_results)
        coint_df = coint_df.merge(robust_df, on=['Asset_1', 'Asset_2'])
        
        # 6. Save Enhanced Results
        report_dir = "output/reports"
        os.makedirs(report_dir, exist_ok=True)
        coint_path = os.path.join(report_dir, f"{self.universe_name}_cointegration_results.csv")
        coint_df.to_csv(coint_path, index=False)
        
        # 7. Spread Plotting for Top Pairs
        # We plot the top 5 by full-period cointegration to allow visual inspection
        top_5_coint = coint_df.sort_values('Coint_P_Value').head(5)
        logger.info(f"Plotting spreads for top 5 cointegrated pairs...")
        self._plot_spread_series(prices, top_5_coint)

        # 8. Plot Normalized Prices for Top Cointegrated
        self._plot_normalized_pairs(prices, list(zip(top_5_coint['Asset_1'], top_5_coint['Asset_2'])))

        # 9. Generate Markdown Ranking Report
        self._generate_ranking_report(coint_df)

        return True

    def _plot_spread_series(self, prices: pd.DataFrame, pairs_df: pd.DataFrame):
        """
        @brief Plots the spread series (Y - gamma*X) for selected pairs.
        """
        output_dir = "output/pairs/spreads"
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
        report_path = f"output/reports/{self.universe_name}_pairs_ranking.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Sort by best overall p-value
        ranked_df = df.sort_values('Coint_P_Value').head(20)
        
        with open(report_path, "w") as f:
            f.write(f"# Pairs Robustness & Ranking Report: {self.universe_name}\n\n")
            f.write("Pairs are ranked by their full-period cointegration. **Robust** pairs are those with $p < 0.05$ in both 2015-2019 and 2020-2025.\n\n")
            
            # Format columns for display
            display_df = ranked_df[[
                'Asset_1', 'Asset_2', 'Correlation', 'Hedge_Ratio', 
                'Coint_P_Value', 'P_Val_2015_2020', 'P_Val_2020_2025', 'Half_Life_Days', 'Is_Robust'
            ]].copy()
            
            # Style
            def style_robust(row):
                return f"**{row['Is_Robust']}**" if row['Is_Robust'] else f"{row['Is_Robust']}"
            
            display_df['Is_Robust'] = display_df.apply(style_robust, axis=1)
            display_df['Coint_P_Value'] = display_df['Coint_P_Value'].map(lambda x: f"**{x:.4f}**" if x < 0.05 else f"{x:.4f}")
            display_df['Half_Life_Days'] = display_df['Half_Life_Days'].map(lambda x: f"{x:.1f}" if not np.isnan(x) else "∞")
            
            f.write(display_df.to_markdown(index=False))
            f.write("\n\n## Plots\n")
            f.write("Visual verification plots can be found in:\n")
            f.write("- **Spread Time Series**: `output/pairs/spreads/`\n")
            f.write("- **Normalized Prices**: `output/pairs/normalized/`\n")

        logger.info(f"Ranking report saved to {report_path}")

    def _plot_normalized_pairs(self, prices: pd.DataFrame, pairs: List[Tuple[str, str]]):
        """
        @brief Generates normalized price plots for the selected pairs.
        """
        output_dir = "output/pairs/normalized"
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
