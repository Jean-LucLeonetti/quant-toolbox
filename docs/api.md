# API Reference

Core components of the Quant research framework.

## Orchestration

### `src.core.pipeline.QuantPipeline`
Central task dispatcher.
- `run(mode: str)`: Dispatches to `stock_analysis` or `universe_build`.

## Research Modules

### `src.research.universe_build.pipeline.UniversePipeline`
Manages the construction of instrument universes.
- `refresh(universes: list)`: Scrapes constituents and updates the metadata DB.

### `src.research.stock_analysis.pipeline`
- `run_stock_analysis(config_path)`: Orchestrates the single-ticker research flow.

## Data & Storage

### `src.data.store.DataStore`
The persistence layer.
- `load/save(...)`: Manages Parquet price caching.
- `upsert_tickers(df)`: Updates the SQLite metadata table.
- `update_universe_membership(tickers, universe)`: Tracks point-in-time membership.
- `query_universe(universe)`: Retrieves active members of a specific universe.

### `src.data.universe`
- `get_sp500_constituents()`: Wikipedia scraper for S&P 500.
- `fetch_ticker_metadata(tickers)`: Batch metadata retrieval from `yfinance`.

### `src.data.fetcher.DataFetcher`
- `fetch_data(ticker, start, end)`: Returns split-adjusted market data (`auto_adjust=True`).
