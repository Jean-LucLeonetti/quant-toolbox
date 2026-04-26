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

## 3. Building Price Panels (data_build)

Before running multi-asset research (like pairs trading), you need to generate a clean, aligned price panel for your target universe.

```bash
python main.py data_build
```
This command:
1. Loads all tickers from the universe specified in `input/configuration.yaml`.
2. Validates data quality (checking for extreme price moves/gaps).
3. Aligns daily adjusted close prices into a single Parquet panel.
4. Generates a **Data Quality Report** in `output/reports/`.

## 4. Pairs Trading Research (pairs)

The `pairs` mode performs an end-to-end statistical arbitrage analysis.

```bash
python main.py pairs
```

### Pipeline Steps:
1. **Cointegration Screening**: Runs the Engle-Granger test on high-correlation pairs.
2. **OU Process Fitting**: Estimates the half-life of mean reversion.
3. **Robustness Validation**: Tests if cointegration persists across sub-periods (split-period testing).
4. **Signal Generation**: Calculates rolling Z-scores and target positions.
5. **Naive Backtest**: Generates equity curves assuming market-neutral execution.

### Diagnostic Outputs:
Results are organized in `output/pairs/` and `output/reports/`:
- **`normalized/`**: "Eyeball test" plots of price co-movement.
- **`spreads/`**: Time-series plots of the OLS residuals.
- **`z_scores/`**: Signal verification plots (Z-score + Target Position).
- **`backtests/`**: Cumulative P&L curves with Sharpe ratio annotations.

## 5. Configuration (Schema Validated)

All research runs consume `input/configuration.yaml`. This file is validated against the schema in `schema/config_schema.json`.

---

## Troubleshooting
- **ModuleNotFoundError: jsonschema**: Run `pip install jsonschema` to enable configuration validation.
- **Extreme Price Detected**: Check the data quality report; some tickers (like levered ETFs) may trigger safety filters during building.
