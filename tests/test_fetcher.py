import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.fetcher import DataFetcher

@pytest.fixture
def clean_fetcher():
    """
    @brief Fixture to provide a clean DataFetcher instance.
    @return A DataFetcher instance.
    """
    return DataFetcher()

@patch('src.data.fetcher.DataStore')
@patch('src.data.fetcher.yf.download')
def test_fetch_data_valid_ticker(mock_download, mock_store_class, clean_fetcher):
    """
    @brief Test fetching data for a valid ticker with mocking.
    """
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    
    # Setup mocks
    mock_store = mock_store_class.return_value
    mock_store.load.return_value = None  # Force cache miss
    
    # Mock return value
    mock_df = pd.DataFrame({'Close': [150.0, 155.0]}, index=pd.date_range(start=start_date, periods=2))
    mock_download.return_value = mock_df
    
    # We need to re-initialize or inject the mock store if clean_fetcher was created before this patch
    clean_fetcher.store = mock_store
    
    df = clean_fetcher.fetch_data(ticker, start_date, end_date)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    mock_download.assert_called_once_with(ticker, start=start_date, end=end_date, auto_adjust=True)
    mock_store.save.assert_called_once()

@patch('src.data.fetcher.DataStore')
@patch('src.data.fetcher.yf.download')
def test_fetch_data_invalid_ticker(mock_download, mock_store_class, clean_fetcher):
    """
    @brief Test fetching data for an invalid ticker with mocking.
    """
    ticker = "INVALID"
    mock_store = mock_store_class.return_value
    mock_store.load.return_value = None
    clean_fetcher.store = mock_store
    
    mock_download.return_value = pd.DataFrame()
    
    df = clean_fetcher.fetch_data(ticker, "2023-01-01", "2023-01-10")
    
    assert df is None

@patch('src.data.fetcher.DataStore')
@patch('src.data.fetcher.yf.download')
def test_fetch_data_exception(mock_download, mock_store_class, clean_fetcher):
    """
    @brief Test fetching data when an exception occurs.
    """
    mock_store = mock_store_class.return_value
    mock_store.load.return_value = None
    clean_fetcher.store = mock_store
    
    mock_download.side_effect = Exception("Connection error")
    
    df = clean_fetcher.fetch_data("AAPL", "2023-01-01", "2023-01-10")
    
    assert df is None
