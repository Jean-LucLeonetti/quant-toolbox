import sys
from src.core.config import Config
from src.data.fetcher import DataFetcher
from src.research.stock_analysis.plotter import StockPlotter
from src.core.logger import setup_logger

logger = setup_logger(__name__)

def run_stock_analysis(config_path: str = "input/configuration.yaml"):
    """
    @brief Orchestrates the single stock analysis pipeline.
    
    @param config_path Path to the configuration file.
    """
    logger.info("Initializing Stock Analysis Pipeline...")

    # 1. Load configuration
    try:
        config = Config.load(config_path)
        ticker = config.data.ticker
        start_date = config.data.start_date
        end_date = config.data.end_date
        logger.info(f"Analysis target: {ticker} ({start_date} to {end_date})")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False

    # 2. Fetch data
    fetcher = DataFetcher()
    data = fetcher.fetch_data(ticker, start_date, end_date)
    
    if data is None or data.empty:
        logger.error(f"No data available for {ticker}. Skipping plot.")
        return False

    # 3. Plot data
    plotter = StockPlotter()
    save_path = plotter.plot_price(data, ticker)
    
    if save_path:
        logger.info(f"Stock analysis for {ticker} completed successfully.")
        return True
    else:
        logger.error(f"Failed to generate analysis plot for {ticker}.")
        return False
