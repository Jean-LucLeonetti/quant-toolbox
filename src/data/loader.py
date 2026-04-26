import pandas as pd
from typing import Optional
from src.data.store import DataStore
from src.core.logger import setup_logger

logger = setup_logger(__name__)

def load_universe_prices(universe_name: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    @brief Loads the aligned price panel for a given universe.
    
    @param universe_name The name of the universe (e.g., 'sp500_utilities').
    @param start_date Optional start date in YYYY-MM-DD.
    @param end_date Optional end date in YYYY-MM-DD.
    
    @return A DataFrame where columns are tickers and values are adjusted close prices.
    """
    store = DataStore()
    df = store.load_processed(universe_name)
    
    if df is None:
        logger.warning(f"No processed panel found for universe: {universe_name}")
        return None
    
    if start_date or end_date:
        df = df.loc[start_date:end_date]
        
    return df
