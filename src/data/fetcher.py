import yfinance as yf
import pandas as pd
from typing import Optional
from src.core.logger import setup_logger

# Use centralized logger
logger = setup_logger(__name__)

class DataFetcher:
    """
    @brief A class to fetch historical financial data using yfinance.
    """

    def __init__(self):
        pass

    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        @brief Fetches historical data for a given ticker from start_date to end_date.

        @param ticker The ticker symbol to fetch (e.g., 'AAPL').
        @param start_date The start date in 'YYYY-MM-DD' format.
        @param end_date The end date in 'YYYY-MM-DD' format.

        @return A pandas DataFrame containing historical data, or None if an error occurs.
        """
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        
        try:
            # Download data using yfinance
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for ticker {ticker} in the specified range.")
                return None
            
            logger.info(f"Successfully fetched {len(data)} rows for {ticker}.")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            return None

