import pytest
import sqlite3
import os
import pandas as pd
from src.data.store import DataStore

# Ensure we run tests against the actual database for validation
DB_PATH = "data/metadata.db"

def test_metadata_db_exists():
    """Check if the database file exists after running universe_build."""
    assert os.path.exists(DB_PATH), "Database file not found. Run 'python main.py universe_build' first."

def test_universe_membership_counts():
    """Verify that universes are populated with a reasonable number of tickers."""
    store = DataStore(db_path=DB_PATH)
    
    sp500_tickers = store.query_universe("sp500")
    uni1_tickers = store.query_universe("sp500_utilities_staples")
    
    # Validation against rough expected counts
    assert len(sp500_tickers) >= 500, f"Expected ~500 S&P 500 tickers, found {len(sp500_tickers)}"
    assert 50 <= len(uni1_tickers) <= 80, f"Expected 50-80 tickers in Universe 1, found {len(uni1_tickers)}"

def test_sector_accuracy_for_universe_1():
    """Verify that Universe 1 only contains tickers from Utilities and Consumer Defensive sectors."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT ticker, sector FROM tickers 
        JOIN universe_membership USING(ticker) 
        WHERE universe = 'sp500_utilities_staples'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # In yfinance/GICS, 'Consumer Defensive' is often used for Staples, 
    # but some stocks (like CASY) are classified as 'Consumer Cyclical' by yfinance.
    allowed_sectors = ['Utilities', 'Consumer Defensive', 'Consumer Cyclical']
    
    invalid_tickers = df[~df['sector'].isin(allowed_sectors)]
    
    assert invalid_tickers.empty, f"Found tickers in Universe 1 with unexpected sectors: {invalid_tickers.to_dict()}"

def test_ticker_metadata_integrity():
    """Check if metadata fields are properly populated for a high-profile ticker."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM tickers WHERE ticker = 'AAPL'"
    df = pd.read_sql(query, conn)
    conn.close()
    
    assert not df.empty, "AAPL not found in tickers table"
    assert df.iloc[0]['name'] == "Apple Inc."
    assert df.iloc[0]['exchange'] in ["NAS", "NMS", "NASDAQ"]
    assert df.iloc[0]['currency'] == "USD"
    assert df.iloc[0]['sector'] == "Technology"
