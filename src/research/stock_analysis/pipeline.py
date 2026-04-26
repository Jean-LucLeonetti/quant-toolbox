import sys
import pandas as pd
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
    ticker_info = fetcher.get_ticker_info(ticker)
    
    if data is None or data.empty:
        logger.error(f"No data available for {ticker}. Skipping analysis.")
        return False

    # Create ticker-specific output folder
    import os
    output_dir = os.path.join("output", ticker)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Generate Markdown Report
    report_path = os.path.join(output_dir, f"{ticker}_summary.md")
    try:
        with open(report_path, "w") as f:
            f.write(f"# Stock Analysis Report: {ticker}\n\n")
            
            # Basic Info from yfinance
            name = ticker_info.get('longName', ticker)
            sector = ticker_info.get('sector', 'N/A')
            industry = ticker_info.get('industry', 'N/A')
            summary = ticker_info.get('longBusinessSummary', 'No description available.')
            
            f.write(f"## {name}\n")
            f.write(f"**Sector:** {sector}  \n")
            f.write(f"**Industry:** {industry}  \n\n")
            f.write(f"### Business Summary\n{summary}\n\n")
            
            # Technical Metrics
            f.write("### Technical Metrics\n")
            last_price = data['Close'].iloc[-1]
            if isinstance(last_price, pd.Series):
                last_price = last_price.iloc[0]
                
            volatility = data['Close'].pct_change().std() * (252**0.5)
            if isinstance(volatility, pd.Series):
                volatility = volatility.iloc[0]
                
            f.write(f"| Metric | Value |\n")
            f.write(f"| --- | --- |\n")
            f.write(f"| Last Close | ${last_price:.2f} |\n")
            f.write(f"| Annualized Volatility | {volatility:.2%} |\n")
            f.write(f"| Market Cap | ${ticker_info.get('marketCap', 0):,.0f} |\n")
            f.write(f"| 52 Week High | ${ticker_info.get('fiftyTwoWeekHigh', 0):.2f} |\n")
            f.write(f"| 52 Week Low | ${ticker_info.get('fiftyTwoWeekLow', 0):.2f} |\n\n")
            
            f.write(f"![{ticker} Price Chart]({ticker}_price.png)\n")
            
        logger.info(f"Summary report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to generate markdown report: {e}")

    # 4. Plot data
    plotter = StockPlotter()
    save_path = plotter.plot_price(data, ticker, output_dir=output_dir)
    
    if save_path:
        logger.info(f"Stock analysis for {ticker} completed successfully.")
        return True
    else:
        logger.error(f"Failed to generate analysis plot for {ticker}.")
        return False
