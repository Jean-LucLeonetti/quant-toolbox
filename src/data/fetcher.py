import yfinance as yf
import pandas as pd
from typing import Optional
from src.core.logger import setup_logger
from src.data.store import DataStore

# Use centralized logger
logger = setup_logger(__name__)

class DataFetcher:
    """
    @brief A class to fetch historical financial data using yfinance with local caching.
    """

    def __init__(self):
        self.store = DataStore()

    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        @brief Fetches historical data for a given ticker from start_date to end_date.
        Checks local cache first, then falls back to yfinance.

        @param ticker The ticker symbol to fetch (e.g., 'AAPL').
        @param start_date The start date in 'YYYY-MM-DD' format.
        @param end_date The end date in 'YYYY-MM-DD' format.

        @return A pandas DataFrame containing historical data, or None if an error occurs.
        """
        # 1. Try to load from cache
        cached_data = self.store.load(ticker, start_date, end_date)
        if cached_data is not None:
            return cached_data

        # 2. Fall back to API
        logger.info(f"Cache miss. Fetching data for {ticker} from {start_date} to {end_date}...")
        
        try:
            # Download data using yfinance with auto_adjust=True to ensure
            # split and dividend adjusted prices are in the 'Close' column.
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            
            if data.empty:
                logger.warning(f"No data found for ticker {ticker} in the specified range.")
                return None
            
            # 3. Save to cache for future use
            self.store.save(data, ticker, start_date, end_date)
            
            logger.info(f"Successfully fetched {len(data)} rows for {ticker}.")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            return None
    def fetch_prices(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        @brief Fetches historical prices for a given ticker.
        Always fetches from API and auto-adjusts.
        """
        logger.info(f"Fetching prices for {ticker} from {start_date} to {end_date}...")
        try:
            # We want both standard OHLC and adjusted if possible, 
            # but yfinance auto_adjust=True replaces 'Close' with adjusted close.
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            if data.empty:
                logger.warning(f"No data found for ticker {ticker}.")
                return None
            return data
        except Exception as e:
            logger.error(f"Failed to fetch prices for {ticker}: {e}")
            return None
    def get_ticker_info(self, ticker: str) -> dict:
        """
        @brief Fetches descriptive metadata for a ticker using yfinance.
        """
        try:
            t = yf.Ticker(ticker)
            return t.info
        except Exception as e:
            logger.error(f"Failed to fetch info for {ticker}: {e}")
            return {}
