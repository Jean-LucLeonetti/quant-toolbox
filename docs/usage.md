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
- **`backtests/`**: Cumulative P&L curves for individual pairs and the **Aggregated Portfolio**.

## 5. Advanced Research Configuration

The behavior of the `pairs` pipeline can be heavily tuned in `input/configuration.yaml`.

### Portfolio Aggregation (`portfolio_size`)
- Set `portfolio_size` (e.g., `30`) to test an equally-weighted portfolio of the top $N$ cointegrated pairs.
- By leveraging the Law of Large Numbers, trading a basket of loosely cointegrated pairs often achieves lower drawdowns and a higher aggregate Sharpe Ratio than trading a single "perfect" pair.

### Hedge Ratio Modes (`hedge_mode`)
- **`static_ols`**: Best for stationary assets (Utilities). Uses a fixed ratio for the whole period.
- **`kalman_filter`**: Best for drifting assets (ETFs). Adaptively updates $\beta_t$ daily to capture structural shifts.

### Cointegration Tests (`coint_mode`)
- **`engle_granger`**: Standard two-step OLS approach. Fast and intuitive.
- **`johansen`**: Vector Autoregression based. Direction-agnostic and more powerful for filtering out borderline spurious correlations.

### Regime & Risk Controls
- **`excluded_periods`**: Exclude specific date ranges (like the 2020 COVID shock) to prevent statistical distortion.
- **`regime_filter`**: Set to `true` to enable a volatility-based circuit breaker. The strategy will sit out if rolling spread volatility exceeds the 90th percentile.

---

## Troubleshooting
- **Johansen Results seem high**: The Johansen test reports Trace Statistics, not p-values. Compare against critical values (typically ~15.5 for significance at 5%).
- **Signal Absorption**: If using the Kalman Filter on stationary assets, the filter may "eat" the mean-reversion signal by over-adapting to noise.
