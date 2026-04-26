import os
import pandas as pd
from typing import Optional
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class DataStore:
    """
    @brief A caching layer for financial data using Parquet files.
    """
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, ticker: str, start_date: str, end_date: str) -> str:
        """Generates a unique filename for the given parameters."""
        filename = f"{ticker}_{start_date}_{end_date}.parquet"
        return os.path.join(self.cache_dir, filename)

    def load(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        @brief Loads data from the local cache if available.
        """
        path = self._get_cache_path(ticker, start_date, end_date)
        if os.path.exists(path):
            logger.info(f"Loading cached data for {ticker} from {path}")
            return pd.read_parquet(path)
        return None

    def save(self, data: pd.DataFrame, ticker: str, start_date: str, end_date: str):
        """
        @brief Saves data to the local cache.
        """
        if data is None or data.empty:
            return
            
        path = self._get_cache_path(ticker, start_date, end_date)
        logger.info(f"Caching data for {ticker} to {path}")
        data.to_parquet(path)
