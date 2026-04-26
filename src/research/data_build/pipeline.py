import os
import pandas as pd
import numpy as np
from typing import List, Dict
from src.core.logger import setup_logger
from src.data.store import DataStore
from src.data.fetcher import DataFetcher
from src.data.validation import validate_ticker_data, ValidationResult

logger = setup_logger(__name__)

class DataBuildPipeline:
    """
    @brief Orchestrates the building of a validated price panel.
    """
    def __init__(self, config: Dict = None):
        self.store = DataStore()
        self.fetcher = DataFetcher()
        self.config = config or {}

    def build(self, universe_name: str, start: str, end: str):
        """
        @brief Main entry point for the data build process.
        """
        logger.info(f"Building data panel for universe: {universe_name} ({start} to {end})")
        
        # 1. Resolve Universe
        tickers = self.store.query_universe(universe_name)
        if not tickers:
            logger.error(f"No tickers found for universe: {universe_name}")
            return False

        logger.info(f"Found {len(tickers)} tickers in universe.")

        # 2. Fetch and Refresh Raw Prices
        validation_results = []
        valid_tickers = []
        
        for ticker in tickers:
            # Check freshness (max 7 days)
            if not self.store.has_fresh(ticker, max_age_days=7):
                df = self.fetcher.fetch_prices(ticker, start, end)
                if df is not None:
                    self.store.save_raw(ticker, df)
            
            # Load raw data (regardless of whether it was just fetched or was already there)
            df = self.store.load_raw(ticker)
            
            # 3. Validate
            result = validate_ticker_data(ticker, df, start, end)
            validation_results.append(result)
            
            if result.is_valid:
                valid_tickers.append(ticker)
            else:
                logger.warning(f"Ticker {ticker} failed validation: {result.errors}")

        logger.info(f"Validation complete. {len(valid_tickers)}/{len(tickers)} tickers are valid.")

        if not valid_tickers:
            logger.error("No valid tickers found. Aborting panel build.")
            return False

        # 4. Align into a Panel
        logger.info("Aligning valid tickers into a panel...")
        ticker_series = {}
        for ticker in valid_tickers:
            df = self.store.load_raw(ticker)
            # Ensure index is datetime and filtered for the range
            df.index = pd.to_datetime(df.index)
            # We use 'Close' as it should be the adjusted close from yfinance auto_adjust=True
            # However, yfinance sometimes returns a MultiIndex if multiple tickers were requested, 
            # but here we fetch one by one. Check if 'Close' is a column.
            if 'Close' in df.columns:
                ticker_series[ticker] = df['Close']
            else:
                logger.warning(f"Ticker {ticker} missing 'Close' column? Columns: {df.columns}")

        panel = pd.concat(ticker_series, axis=1)
        panel = panel.sort_index()

        # Handle range alignment
        panel = panel.loc[start:end]

        # Policy: Strict inner join (drop rows with any NaN)
        initial_rows = len(panel)
        panel = panel.dropna()
        dropped_rows = initial_rows - len(panel)
        
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows due to NaN values in alignment.")

        # 5. Save Processed Panel
        self.store.save_processed(universe_name, panel)

        # 6. Generate Quality Report
        self._generate_report(universe_name, validation_results, panel)

        return True

    def _generate_report(self, universe_name: str, results: List[ValidationResult], panel: pd.DataFrame):
        """
        @brief Generates a markdown data quality report.
        """
        report_path = f"output/reports/{universe_name}_quality_report.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, "w") as f:
            f.write(f"# Data Quality Report: {universe_name}\n\n")
            f.write(f"**Universe:** {universe_name}  \n")
            f.write(f"**Date Range:** {panel.index.min().date()} to {panel.index.max().date()}  \n")
            f.write(f"**Ticker Count:** {panel.shape[1]} valid tickers  \n")
            f.write(f"**Row Count:** {panel.shape[0]} trading days  \n\n")
            
            f.write("## Validation Summary\n\n")
            f.write("| Ticker | Valid | Errors | Warnings |\n")
            f.write("| --- | --- | --- | --- |\n")
            for r in results:
                err_str = ", ".join(r.errors) if r.errors else "None"
                warn_str = ", ".join(r.warnings) if r.warnings else "None"
                f.write(f"| {r.ticker} | {'✅' if r.is_valid else '❌'} | {err_str} | {warn_str} |\n")
            
            f.write("\n## Correlation Matrix (Top 5 Pairs)\n\n")
            if not panel.empty:
                returns = panel.pct_change().dropna()
                corr = returns.corr()
                # Get upper triangle of correlation matrix
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                # Stack and sort
                top_corr = upper.unstack().dropna().sort_values(ascending=False).head(10)
                
                f.write("| Asset 1 | Asset 2 | Correlation |\n")
                f.write("| --- | --- | --- |\n")
                for (a1, a2), val in top_corr.items():
                    f.write(f"| {a1} | {a2} | {val:.4f} |\n")
            
            f.write("\n## Panel Statistics\n\n")
            stats = panel.describe().transpose()
            f.write(stats.to_markdown())

        logger.info(f"Data quality report saved to {report_path}")
