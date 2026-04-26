# Quant Toolbox: Statistical Arbitrage Framework

A robust, research-oriented pipeline for discovering and validating pairs trading strategies across diverse asset universes. Built to handle everything from parameter optimization to out-of-sample stress testing.

## Features
- **Honest Walk-Forward Validation**: Rolling train/test engine that eliminates look-ahead bias and gap leaks. It measures "Expectation vs Reality" by comparing training-window performance to out-of-sample execution.
- **Dynamic Hedge Modes**: Seamless switching between Static OLS and recursive Kalman Filter for adaptive hedge ratios.
- **Trade Diagnostics**: Automatic tracking of trade counts, win rates, and degradation percentages to detect overfitting.
- **Experiment Leaderboard**: Centralized logging of all runs in `experiments_log.csv` for cross-universe comparison.
- **Structural Alpha Universes**: Pre-configured SQLite universes for S&P 500, Sector ETFs, Country ETFs, and more.

## Quick Start

1. **Environment Setup**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run a Single Universe**:
   ```bash
   # Executes the configuration defined in input/configuration.yaml
   python3 main.py pairs
   ```

3. **Run a Grid Experiment**:
   ```bash
   # Runs a multi-universe, multi-parameter grid search and updates the leaderboard
   python3 scripts/run_experiments.py
   ```

## Project Structure
- `src/core/`: Base pipeline and configuration classes.
- `src/research/pairs/`: Core logic for cointegration, signal generation, and walk-forward validation.
- `input/`: Configuration YAMLs and SQLite universe definitions.
- `output/runs/`: Per-run diagnostic plots, reports, and the cross-experiment leaderboard.
- `scripts/`: Batch utilities for large-scale research.

## Documentation
For detailed guides on configuration and methodology:
- [Usage Guide](docs/usage.md)
- [Research Philosophy](docs/research_philosophy.md)
