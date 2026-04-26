# Quant Toolbox

A professional-grade modular Python framework for financial research, algorithmic trading, and pairs analysis.

## Key Features
- **Sophisticated Data Pipeline**: Automatic split/dividend adjustments and Parquet-based local caching.
- **Universe Management**: SQLite-backed metadata store with point-in-time membership tracking.
- **Advanced Cointegration**: Supports **Engle-Granger** and direction-agnostic **Johansen** tests.
- **Dynamic Hedging**: Choose between **Static OLS** or adaptive **Kalman Filter** hedge ratios.
- **Regime Awareness**: Built-in logic for **COVID-period exclusions** and **Volatility-based regime filters**.
- **Execution Backtest**: High-fidelity signal generation with $|Z|>2$ entry and convergence targets.

## Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Build your research database (S&P 500 metadata)
python main.py universe_build

# 3. Pull aligned price panels for your target universe
python main.py data_build

# 4. Run the full pairs trading exploration pipeline (Stats -> Backtest)
python main.py pairs
```

## Configuration

The system is controlled via `input/configuration.yaml`. You can tune:
- **Data Universe**: Switch between `sp500_utilities`, `etf_pairs`, etc.
- **Time Window**: Define `start_date` and `end_date` for the backtest.
- **Strategy Parameters**: Tune `z_window` (60d default), `z_entry` (2.0σ), and `z_exit` (0.5σ).

Validation is enforced by the schema in `schema/config_schema.json`.

## Project Structure
- `src/research/pairs/`: Statistical testing, signal generation, and backtesting.
- `output/reports/`: Markdown ranking reports and data quality summaries.
- `output/pairs/`: Diagnostic plots (Spreads, Z-scores, Equity Curves).
- `input/`: User configuration and YAML schema.
- `src/data/`: Data ingestion, caching, and database management.

## Documentation
- [Usage Guide](docs/usage.md)
- [Architecture & Design](docs/architecture.md)

## License
MIT
