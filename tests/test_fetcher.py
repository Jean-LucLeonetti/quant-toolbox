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

@patch('src.data.fetcher.yf.download')
def test_fetch_data_valid_ticker(mock_download, clean_fetcher):
    """
    @brief Test fetching data for a valid ticker with mocking.
    """
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    
    # Mock return value
    mock_df = pd.DataFrame({'Close': [150.0, 155.0]}, index=pd.date_range(start=start_date, periods=2))
    mock_download.return_value = mock_df
    
    df = clean_fetcher.fetch_data(ticker, start_date, end_date)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    mock_download.assert_called_once_with(ticker, start=start_date, end=end_date)

@patch('src.data.fetcher.yf.download')
def test_fetch_data_invalid_ticker(mock_download, clean_fetcher):
    """
    @brief Test fetching data for an invalid ticker with mocking.
    """
    ticker = "INVALID"
    mock_download.return_value = pd.DataFrame()
    
    df = clean_fetcher.fetch_data(ticker, "2023-01-01", "2023-01-10")
    
    assert df is None

@patch('src.data.fetcher.yf.download')
def test_fetch_data_exception(mock_download, clean_fetcher):
    """
    @brief Test fetching data when an exception occurs.
    """
    mock_download.side_effect = Exception("Connection error")
    
    df = clean_fetcher.fetch_data("AAPL", "2023-01-01", "2023-01-10")
    
    assert df is None
