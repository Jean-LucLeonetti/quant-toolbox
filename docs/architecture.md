# System Architecture

The Quant Toolbox is designed as a modular, extensible framework for financial research and algorithmic trading. It follows a decoupled architecture where data management, orchestration, and research logic are strictly separated.

## Core Components

### 1. Central Pipeline (`src/core/pipeline.py`)
The `QuantPipeline` class is the central orchestrator. It manages the high-level execution flow and dispatches tasks to specific research modules (e.g., `stock_analysis`, `universe_build`).

### 2. Data Layer (`src/data/`)
- **Fetcher (`fetcher.py`)**: Wraps `yfinance` to retrieve market data with split and dividend adjustments (`auto_adjust=True`).
- **Store (`store.py`)**: Manages persistent storage.
    - **Parquet Cache**: Fast storage for historical price data to minimize API calls.
    - **SQLite Metadata**: Stores ticker details (sector, market cap) and point-in-time universe membership.
- **Universe (`universe.py`)**: Logic for discovering and filtering instrument lists (e.g., S&P 500 scraping).

### 3. Research Layer (`src/research/`)
This layer contains independent modules for different trading strategies or research tasks:
- **`stock_analysis/`**: Tools for individual ticker research and visualization.
- **`universe_build/`**: Pipelines for constructing and maintaining research universes (e.g., S&P 500 Utilities).
- **`pairs/`**: (Future) Cointegration and correlation-based pairs trading logic.

## Data Flow

1. **Initialization**: `main.py` dispatches a `mode` (e.g., `universe_build`) to the `QuantPipeline`.
2. **Metadata Population**: `UniversePipeline` scrapes constituents, fetches metadata via `DataFetcher`, and stores it in the SQLite database.
3. **Execution**: A research module (like `stock_analysis`) queries the metadata store to define its target universe.
4. **Data Acquisition**: The module requests historical data. `DataFetcher` checks the Parquet cache before hitting the API.
5. **Output**: Analysis results and plots are saved to the `output/` directory.

## Design Philosophy

- **Point-in-Time Integrity**: Ticker membership is tracked by date to prevent survivorship bias in future backtests.
- **Modularity**: Every research strategy is contained within its own package under `src/research/`.
- **Reproducibility**: Environment and data states are preserved to ensure consistent results.
