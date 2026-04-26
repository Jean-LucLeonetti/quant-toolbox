# Usage Guide

This guide explains how to manage data and run research pipelines in the Quant Toolbox.

## 1. Environment Setup

```bash
# Activate the 3.14.3 environment
source .venv/bin/activate
# Install necessary libraries (yfinance, pandas, matplotlib, pyarrow, lxml)
pip install -r requirements.txt
```

## 2. Managing Universes

Before running specialized analysis, you must build your metadata database.

### Build/Refresh Research Universes
```bash
python main.py universe_build
```
This command:
1. Scrapes the current S&P 500 constituents from Wikipedia.
2. Fetches sector and company metadata for all tickers.
3. Automatically identifies **Universe 1** (S&P 500 Utilities and Consumer Staples).
4. Populates the SQLite database at `data/metadata.db`.

## 3. Stock Analysis

To analyze and plot a specific stock:
1. Update `input/configuration.yaml` with the desired ticker and dates.
2. Run the analysis:
```bash
python main.py stock_analysis
```

### Outputs
- **Plots**: Visualizations are saved in `output/` (e.g., `output/AAPL_price.png`).
- **Data Cache**: Raw market data is cached as Parquet files in `data/cache/` to speed up future runs.

## 4. Querying Metadata

You can query the SQLite database directly to explore your universes:
```bash
sqlite3 data/metadata.db "SELECT ticker, sector FROM tickers WHERE sector = 'Utilities';"
```

## Troubleshooting
- **403 Forbidden**: Ensure your internet connection is active; the scraper uses a custom User-Agent to avoid Wikipedia blocks.
- **Missing Data**: If a plot is blank, check `src.data.fetcher` logs to verify if Yahoo Finance returned data for the requested dates.
