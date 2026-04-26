# System Architecture

The Quant Toolbox is designed as a modular, extensible framework for financial data analysis. It follows a decoupled architecture where orchestration is separated from execution.

## Core Components

### 1. Central Pipeline (`src/core/pipeline.py`)
The `QuantPipeline` class is the central orchestrator. It manages the high-level execution flow and determines which analysis module to run based on the user's requirements.

### 2. Configuration Management (`src/core/config.py`)
Handles parsing of YAML configuration files and provides type-safe access to settings via Python dataclasses.

### 3. Data Ingestion (`src/data/fetcher.py`)
A generalized wrapper for financial data sources (currently using `yfinance`). It handles fetching historical market data and provides logging for transparency.

### 4. Logic Modules

#### Stock Analysis (`src/stock_analysis/`)
- **Pipeline**: Manages the specific sequence of operations for a single stock (Fetch -> Analyze -> Plot).
- **Plotter**: Handles the visual representation of data using `matplotlib`.

## Data Flow

1. **Initialization**: `main.py` initializes the `QuantPipeline` with a configuration path.
2. **Configuration**: The pipeline loads settings (ticker, dates) from `configuration.yaml`.
3. **Execution**: The pipeline triggers the appropriate analysis module (e.g., `stock_analysis`).
4. **Data Acquisition**: The specialized module calls the `DataFetcher` to retrieve market data.
5. **Output**: Results (plots, metrics) are generated and saved to the `output/` directory.

## Design Philosophy

- **Modularity**: New components (like Pairs Trading or Backtesting) can be added as self-contained packages.
- **Portability**: Uses virtual environments and relative paths (`${workspaceFolder}`) to ensure code runs consistently across different setups.
- **Stability**: Comprehensive logging and specific python version requirements (Python 3.14+) ensure robust execution.
