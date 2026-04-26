import matplotlib.pyplot as plt
import pandas as pd
import os
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class StockPlotter:
    """
    @brief A class to plot stock price data.
    """
    def __init__(self):
        # Use a clean, modern style
        try:
            plt.style.use('ggplot')
        except:
            pass

    def plot_price(self, data: pd.DataFrame, ticker: str, output_dir: str = "output"):
        """
        @brief Plots the closing price of the stock and saves it to a file.
        
        @param data Pandas DataFrame containing the stock data.
        @param ticker The stock ticker symbol.
        @param output_dir Directory to save the plot.
        """
        if data is None or data.empty:
            logger.error(f"No data available to plot for {ticker}")
            return None

        logger.info(f"Generating price plot for {ticker}...")
        
        # Ensure 'Close' is a Series (extract from MultiIndex if necessary)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                plot_data = data['Close']
            else:
                logger.error(f"No 'Close' column found in MultiIndex for {ticker}")
                return None
        else:
            if 'Close' in data.columns:
                plot_data = data['Close']
            else:
                logger.error(f"No 'Close' column found in data for {ticker}")
                return None

        plt.figure(figsize=(12, 7))
        plt.plot(plot_data.index, plot_data.values, label='Close Price', color='#2ca02c', linewidth=2)
        plt.fill_between(plot_data.index, plot_data.values.flatten(), alpha=0.15, color='#2ca02c')
        
        plt.title(f"{ticker} Historical Price Analysis", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (USD)", fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Format the plot
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{ticker}_price.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Plot successfully saved to {save_path}")
        return save_path
