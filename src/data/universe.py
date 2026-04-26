import pandas as pd
import yfinance as yf
from src.core.logger import setup_logger
from typing import List, Dict, Any

logger = setup_logger(__name__)

def get_sp500_constituents() -> pd.DataFrame:
    """
    @brief Scrapes the current S&P 500 constituents from Wikipedia.
    @return A DataFrame containing ticker symbols and metadata (Security, GICS Sector, etc.)
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        logger.info("Scraping S&P 500 constituents from Wikipedia...")
        import requests
        from io import StringIO
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        # Clean up ticker symbols (some have periods instead of hyphens)
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        return df
    except Exception as e:
        logger.error(f"Failed to scrape S&P 500 constituents: {e}")
        return pd.DataFrame()

def fetch_ticker_metadata(tickers: List[str]) -> pd.DataFrame:
    """
    @brief Fetches detailed metadata for a list of tickers via yfinance.
    @param tickers List of ticker symbols.
    @return A DataFrame with metadata records.
    """
    records = []
    logger.info(f"Fetching metadata for {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers):
        try:
            if i % 10 == 0 and i > 0:
                logger.info(f"Progress: {i}/{len(tickers)}...")
                
            t = yf.Ticker(ticker)
            info = t.info
            
            records.append({
                'ticker': ticker,
                'name': info.get('longName'),
                'exchange': info.get('exchange'),
                'currency': info.get('currency'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'avg_volume': info.get('averageVolume'),
                'is_active': True
            })
        except Exception as e:
            logger.warning(f"Failed to fetch metadata for {ticker}: {e}")
            
    return pd.DataFrame(records)

def filter_universe_by_sector(df: pd.DataFrame, sectors: List[str]) -> List[str]:
    """
    @brief Filters a constituency DataFrame by GICS sectors.
    @param df The constituents DataFrame (from get_sp500_constituents).
    @param sectors List of sectors to include (e.g., ['Utilities', 'Consumer Staples']).
    @return List of matching ticker symbols.
    """
    mask = df['GICS Sector'].isin(sectors)
    return df[mask]['Symbol'].tolist()
