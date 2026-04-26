import logging
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from src.core.logger import setup_logger

logger = setup_logger(__name__)

# Add file handler for data quality specifically
dq_log_path = "data/data_quality.log"
os.makedirs("data", exist_ok=True)
fh = logging.FileHandler(dq_log_path)
fh.setLevel(logging.WARNING)
fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

@dataclass
class ValidationResult:
    ticker: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    nan_count: int = 0

def validate_ticker_data(ticker: str, df: pd.DataFrame, expected_start: str, expected_end: str) -> ValidationResult:
    """
    @brief Validates price data for a single ticker.
    """
    errors = []
    warnings = []
    
    if df is None or df.empty:
        return ValidationResult(ticker=ticker, is_valid=False, errors=["No data found"])

    # Check coverage
    actual_start = df.index.min().strftime('%Y-%m-%d')
    actual_end = df.index.max().strftime('%Y-%m-%d')
    
    if actual_start > expected_start:
        errors.append(f"Insufficient coverage: starts at {actual_start}, expected {expected_start}")
    
    # Check for NaNs in 'Close'
    nan_count = df['Close'].isna().sum()
    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN values in Close price")

    # Check for internal gaps (> 5 trading days)
    # Note: This assumes business days. 
    # A more robust check would use a trading calendar, but diff > 7 days is a safe proxy for week-long gaps.
    date_diffs = df.index.to_series().diff().dt.days
    max_gap = date_diffs.max()
    if max_gap > 7:
        errors.append(f"Internal gap detected: max gap of {max_gap} days")

    # Check for stale prices (>= 5 consecutive identical closes)
    # Identical closes can happen in low volume, but for S&P 500 utilities it's suspicious.
    stale_mask = (df['Close'] == df['Close'].shift(1))
    # Simple check for sequences
    consecutive_stale = stale_mask.rolling(window=5).sum()
    if (consecutive_stale >= 4).any(): # 4 True in a row means 5 identical values
        warnings.append("Stale prices detected (5+ consecutive identical closes)")

    # Check for outliers (> 50% single day move)
    returns = df['Close'].pct_change().abs()
    outliers = returns[returns > 0.5]
    if not outliers.empty:
        errors.append(f"Extreme price moves detected (>50%): {outliers.index.tolist()}")

    is_valid = len(errors) == 0
    
    return ValidationResult(
        ticker=ticker,
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        start_date=actual_start,
        end_date=actual_end,
        nan_count=nan_count
    )
