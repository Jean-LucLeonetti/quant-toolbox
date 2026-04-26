# Quant Toolbox

A professional-grade modular Python framework for financial research, algorithmic trading, and pairs analysis.

## Key Features
- **Sophisticated Data Pipeline**: Automatic split/dividend adjustments and Parquet-based local caching.
- **Universe Management**: SQLite-backed metadata store with point-in-time membership tracking (prevents survivorship bias).
- **Research-Centric Design**: Modular `src/research` structure for developing independent trading strategies.
- **Optimized Environment**: Fully compatible and tested with Python 3.14.3.

## Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Build your research database (S&P 500 metadata)
python main.py universe_build

# 3. Run a stock analysis
python main.py stock_analysis
```

## Project Structure
- `docs/`: Detailed guides and architecture specs.
- `src/research/`: Your trading strategies and analysis modules.
- `src/data/`: Data ingestion, caching, and database management.
- `src/core/`: Orchestration and configuration.

## Documentation
- [System Architecture](docs/architecture.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api.md)

## License
MIT
