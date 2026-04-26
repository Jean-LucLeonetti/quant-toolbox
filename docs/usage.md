# Usage Guide

This guide explains how to configure and run the Quant Toolbox.

## Prerequisites

- **Python 3.14+**: The environment is highly optimized for Python 3.14.3.
- **Virtual Environment**: We use a `.venv` directory for dependency isolation.

## Setup

1. **Activate Environment**:
   ```bash
   source .venv/bin/activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

All analysis parameters are stored in `input/configuration.yaml`.

Example configuration:
```yaml
data:
  ticker: "AAPL"       # Stock symbol
  start_date: "2020-01-01"
  end_date: "2025-12-31"
```

## Running the Analysis

To run the default single stock analysis pipeline:

```bash
python main.py
```

### Outputs
- **Plots**: Visualizations are saved in the `output/` directory (e.g., `output/AAPL_price.png`).
- **Logs**: Execution details are printed to the console via the centralized logging system.

## Troubleshooting

- **Invalid Interpreter**: Ensure your VS Code is pointing to `${workspaceFolder}/.venv/bin/python`.
- **Data Fetching Errors**: Check your internet connection and verify that the ticker symbol is valid on Yahoo Finance.
