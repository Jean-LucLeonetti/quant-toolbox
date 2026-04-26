import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
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

        # 4. Step B: Cointegration Testing (Engle-Granger)
        logger.info("Performing Engle-Granger cointegration tests...")
        coint_results = []
        for (a, b), corr in candidates.items():
            # coint returns (t-stat, p-value, critical_values)
            score, pvalue, _ = coint(prices[a], prices[b])
            coint_results.append({
                'Asset_1': a,
                'Asset_2': b,
                'Correlation': corr,
                'Coint_T_Stat': score,
                'Coint_P_Value': pvalue,
                'Is_Cointegrated': pvalue < 0.05
            })
            
        coint_df = pd.DataFrame(coint_results)
        report_dir = "output/reports"
        os.makedirs(report_dir, exist_ok=True)
        coint_path = os.path.join(report_dir, f"{self.universe_name}_cointegration_results.csv")
        coint_df.to_csv(coint_path, index=False)
        logger.info(f"Cointegration results saved to {coint_path}")

        # 5. Step C: Eyeball top 5 (by Cointegration p-value)
        top_5_coint = coint_df.sort_values('Coint_P_Value').head(5)
        logger.info("Top 5 cointegrated pairs:")
        for _, row in top_5_coint.iterrows():
            logger.info(f"  {row['Asset_1']} - {row['Asset_2']}: p={row['Coint_P_Value']:.4f} (corr={row['Correlation']:.4f})")

        self._plot_normalized_pairs(prices, list(zip(top_5_coint['Asset_1'], top_5_coint['Asset_2'])))

        return True

    def _plot_normalized_pairs(self, prices: pd.DataFrame, pairs: List[Tuple[str, str]]):
        """
        @brief Generates normalized price plots for the selected pairs.
        """
        output_dir = "output/pairs"
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
