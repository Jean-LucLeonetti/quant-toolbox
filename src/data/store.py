import os
import sqlite3
import pandas as pd
from typing import Optional, List
from datetime import date
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class DataStore:
    """
    @brief A caching layer for financial data using Parquet files.
    """
    def __init__(self, cache_dir: str = "data/cache", db_path: str = "data/metadata.db"):
        self.cache_dir = cache_dir
        self.raw_dir = "data/raw"
        self.processed_dir = "data/processed"
        self.db_path = db_path
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database with the required schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tickers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                currency TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                avg_volume REAL,
                is_active BOOLEAN,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Universe membership table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS universe_membership (
                ticker TEXT,
                universe TEXT,
                start_date DATE,
                end_date DATE,
                PRIMARY KEY (ticker, universe, start_date)
            )
        """)
        
        conn.commit()
        conn.close()

    def upsert_tickers(self, df: pd.DataFrame):
        """Insert or update ticker metadata."""
        if df.empty:
            return
            
        conn = sqlite3.connect(self.db_path)
        # We use a temporary table to handle the upsert (SQLite 3.24+ has ON CONFLICT, but this is safer)
        df.to_sql('tickers_tmp', conn, if_exists='replace', index=False)
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO tickers (ticker, name, exchange, currency, sector, industry, market_cap, avg_volume, is_active)
            SELECT ticker, name, exchange, currency, sector, industry, market_cap, avg_volume, is_active FROM tickers_tmp
        """)
        
        cursor.execute("DROP TABLE tickers_tmp")
        conn.commit()
        conn.close()
        logger.info(f"Upserted {len(df)} tickers to metadata database.")

    def update_universe_membership(self, tickers: List[str], universe_name: str):
        """Updates membership for a given universe, handling additions and removals."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = date.today().isoformat()
        
        # 1. Mark existing members not in the new list as ended (if not already ended)
        # We use a parameterized approach for safety if the list is small, 
        # but for large lists we use a temporary table for the join/anti-join.
        cursor.execute("CREATE TEMPORARY TABLE new_membership (ticker TEXT)")
        cursor.executemany("INSERT INTO new_membership VALUES (?)", [(t,) for t in tickers])
        
        cursor.execute("""
            UPDATE universe_membership 
            SET end_date = ? 
            WHERE universe = ? 
              AND end_date IS NULL 
              AND ticker NOT IN (SELECT ticker FROM new_membership)
        """, (today, universe_name))
        
        # 2. Insert new members (if they don't already have an active record)
        for ticker in tickers:
            cursor.execute("""
                INSERT OR IGNORE INTO universe_membership (ticker, universe, start_date)
                VALUES (?, ?, ?)
            """, (ticker, universe_name, today))
            
        cursor.execute("DROP TABLE new_membership")
        conn.commit()
        conn.close()
        logger.info(f"Updated membership for universe: {universe_name} (Active: {len(tickers)})")

    def query_universe(self, universe_name: str) -> List[str]:
        """Returns the current constituents of a universe."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT ticker FROM universe_membership WHERE universe = ? AND end_date IS NULL"
        df = pd.read_sql(query, conn, params=(universe_name,))
        conn.close()
        return df['ticker'].tolist()

    def has_fresh(self, ticker: str, max_age_days: int = 7) -> bool:
        """Checks if ticker data in raw storage is fresh enough."""
        path = os.path.join(self.raw_dir, f"{ticker}.parquet")
        if not os.path.exists(path):
            return False
            
        from datetime import datetime, timedelta
        file_time = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - file_time) < timedelta(days=max_age_days)

    def save_raw(self, ticker: str, df: pd.DataFrame):
        """Saves raw ticker data to a dedicated file."""
        if df is None or df.empty:
            return
        path = os.path.join(self.raw_dir, f"{ticker}.parquet")
        df.to_parquet(path)
        logger.debug(f"Saved raw data for {ticker} to {path}")

    def load_raw(self, ticker: str) -> Optional[pd.DataFrame]:
        """Loads raw ticker data."""
        path = os.path.join(self.raw_dir, f"{ticker}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    def save_processed(self, universe_name: str, df: pd.DataFrame):
        """Saves the aligned panel to processed storage."""
        if df is None or df.empty:
            return
        path = os.path.join(self.processed_dir, f"{universe_name}_panel.parquet")
        df.to_parquet(path)
        logger.info(f"Saved processed panel for {universe_name} to {path}")

    def load_processed(self, universe_name: str) -> Optional[pd.DataFrame]:
        """Loads a processed panel."""
        path = os.path.join(self.processed_dir, f"{universe_name}_panel.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

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
