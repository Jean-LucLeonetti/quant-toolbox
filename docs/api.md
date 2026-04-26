# API Reference

Brief overview of the core classes in the Quant Toolbox.

## `src.core.pipeline.QuantPipeline`
The main orchestrator for the application.
- `__init__(config_path: str)`: Initialize with the path to the YAML config.
- `run(mode: str)`: Execute a specific analysis mode (`stock_analysis`, etc.).

## `src.data.fetcher.DataFetcher`
Wrapper for market data acquisition.
- `fetch_data(ticker, start_date, end_date)`: Returns a `pd.DataFrame` containing historical prices.

## `src.stock_analysis.plotter.StockPlotter`
Visualization engine.
- `plot_price(data, ticker)`: Generates and saves a historical price chart.

## `src.core.config.Config`
Configuration parser.
- `load(path)`: Loads configuration from a specific YAML file.
- `from_dict(data)`: Creates a config object from a dictionary.
