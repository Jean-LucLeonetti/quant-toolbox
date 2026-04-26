import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
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
        stats_path = f"output/reports/{self.universe_name}_pair_stats.csv"
        stats_df = pd.DataFrame({
            'Annualized_Mean_Return': ann_mean,
            'Annualized_Volatility': ann_vol
        })
        stats_df.to_csv(stats_path)
        
        # 3. Find top 5 highly correlated pairs
        # Get upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        top_pairs = upper.unstack().dropna().sort_values(ascending=False).head(5)
        
        logger.info("Top 5 correlated pairs:")
        for (a, b), val in top_pairs.items():
            logger.info(f"  {a} - {b}: {val:.4f}")

        # 4. Step B: Eyeball normalised price plots
        self._plot_normalized_pairs(prices, top_pairs.index.tolist())

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
