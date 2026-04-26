# Usage Guide: Statistical Arbitrage Pipeline

This guide outlines how to configure and execute pairs trading research experiments.

## 1. Defining the Universe
The pipeline pulls ticker data from the SQLite database defined in `input/configuration.yaml` under `data.db_path`.
Select a universe by updating `data.universe_name`. Available options include:
- `etf_sector` (US Sector ETFs)
- `etf_country` (International Equity ETFs)
- `sp500` (Individual S&P 500 components)

## 2. Running an Experiment
To execute a standard run:
```bash
python3 main.py pairs
```

## 3. Configuration Breakdown
All strategy parameters live in `input/configuration.yaml`. Key settings:

### Signal Parameters
- `z_entry`: Standard deviation threshold for trade entry (e.g., 2.0).
- `z_exit`: Threshold for trade closing (e.g., 0.5).
- `regime_filter`: (True/False) Enables a volatility circuit-breaker based on historical spread variance.

### Walk-Forward Validation Settings
The pipeline uses a rolling window to simulate real-world trading logic. 

- `wf_train_months`: (default: 36) The lookback window used for pair selection and parameter fitting.
- `wf_test_months`: (default: 6) The out-of-sample period where signals are executed blindly.

**The "Honest Degradation" Metric**
The leaderboard calculates `WF_Degradation_Pct` by comparing:
1. **Expectation (IS)**: The average Sharpe achieved across all training windows.
2. **Reality (OOS)**: The total Sharpe of the continuous out-of-sample return series.

A degradation of -30% means that your live performance was 30% lower than your training-sample expectation—a typical and healthy result. A positive degradation usually indicates a structural change in the market or a potential leak in your validation logic.

## 4. Analyzing Results
Each run generates a timestamped folder in `output/runs/` containing:
- **`reports/`**: Markdown ranking reports and CSV cointegration results.
- **`pairs/backtests/`**: Individual pair equity curves and the **Aggregated Portfolio** curve.
- **`experiments_log.csv`**: The central leaderboard comparing all historical runs.

## 5. Statistical Diagnostics
The `experiments_log.csv` includes detailed tracking to ensure research honesty:
- **Avg_Trades_IS/OOS**: Compares trade frequency to detect "Signal Cherry-Picking".
- **Mean_Half_Life**: Average time (in days) the spread takes to return to its mean.
- **Robust_Pairs**: Count of pairs that passed cointegration in multiple distinct sub-periods (full-sample filter).
