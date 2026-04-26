import pytest
import pandas as pd
from src.data.fetcher import DataFetcher

@pytest.fixture
def clean_fetcher():
    """
    @brief Fixture to provide a clean DataFetcher instance.
    @return A DataFetcher instance.
    """
    return DataFetcher()

def test_fetch_data_valid_ticker(clean_fetcher):
    """
    @brief Test fetching data for a valid ticker.
    """
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    
    df = clean_fetcher.fetch_data(ticker, start_date, end_date)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 0

def test_fetch_data_invalid_ticker(clean_fetcher):
    """
    @brief Test fetching data for an invalid ticker.
    """
    ticker = "INVALID_TICKER_12345"
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    
    df = clean_fetcher.fetch_data(ticker, start_date, end_date)
    
    assert df is None

def test_fetch_data_invalid_dates(clean_fetcher):
    """
    @brief Test fetching data with invalid dates (start after end).
    """
    ticker = "AAPL"
    start_date = "2023-01-10"
    end_date = "2023-01-01"
    
    df = clean_fetcher.fetch_data(ticker, start_date, end_date)
    
    # yfinance might return empty df or handle it, our fetcher returns None if empty
    assert df is None
