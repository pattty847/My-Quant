# This is My Quant - Project Overview

This document provides a quick overview of the "This is My Quant" project structure, key files, and functionality to help understand the system architecture and code organization.

## Project Purpose

This project is a quantitative analysis system for financial markets that:

- Analyzes various markets (crypto, stocks, economic indicators)
- Performs statistical and quantitative analysis
- Identifies patterns preceding market trend changes
- Uses AI to provide market insights
- Supports backtesting of market conditions
- Fetches and processes historical market data

## Core Architecture

The system follows a modular architecture with these main components:

1. **Data Connectors**: Fetch market data from various sources
2. **Data Cache**: Store retrieved data to minimize API requests
3. **Feature Engineering**: Calculate technical indicators and market metrics
4. **Analysis**: Identify market conditions and perform statistical analysis
5. **Backtesting**: Test market conditions against historical data
6. **LLM Integration**: Use language models to analyze and explain market conditions
7. **Interface**: CLI and API interfaces to interact with the system

## Key Files and Directories

### Entry Points
- `main.py` - Main entry point with CLI and API mode selection
- `quant_system/interface/cli.py` - Command-line interface implementation
- `quant_system/interface/api.py` - REST API implementation

### Data Layer
- `quant_system/data/connectors.py` - Market data connectors (currently focused on crypto)
- `quant_system/data/cache.py` - Caching system for market data

### Analysis Layer
- `quant_system/features/technical.py` - Technical indicator calculation
- `quant_system/analysis/market_structure.py` - Market condition identification
- `quant_system/analysis/backtest.py` - Backtesting functionality
- `quant_system/analysis/llm_interface.py` - Language model integration for analysis

### Utilities
- `quant_system/utils.py` - Logging, error handling, and utility functions

### Examples and Documentation
- `examples/` - Example usage scripts
- `docs/` - Documentation and project information

## Command-Line Interface

The system provides these main commands:

1. **analyze**: Analyze a market with technical indicators and conditions
   ```bash
   python main.py cli analyze --symbol BTC/USDT --timeframe 1d --days 365
   ```

2. **symbols**: List available trading pairs
   ```bash
   python main.py cli symbols
   ```

3. **timeframes**: List available timeframes
   ```bash
   python main.py cli timeframes
   ```

4. **backtest**: Backtest specific market conditions
   ```bash
   python main.py cli backtest --conditions "Bullish trend" "Oversold RSI" --symbol BTC/USDT
   ```

5. **history**: Fetch complete historical data with pagination
   ```bash
   python main.py cli history --symbol BTC/USDT --from 2020-01-01 --to 2023-01-01
   ```

## Data Flow

1. User initiates command through CLI or API
2. System checks cache for requested data
3. If needed, fetches data from external sources (exchanges, APIs)
4. Calculates technical indicators and features
5. Performs requested analysis or backtesting
6. Returns results to user through CLI output or API response

## Key Features

### Data Fetching and Caching
- Smart caching system to reduce API calls
- Pagination for retrieving complete historical data
- Support for different timeframes and markets

### Technical Analysis
- Calculation of technical indicators (RSI, MA, etc.)
- Market condition identification
- Pattern recognition

### Backtesting
- Test conditions against historical data
- Calculate forward returns after specific conditions
- Statistical analysis of results

### AI Integration
- LLM-based market analysis
- Natural language summaries of market conditions
- Pattern recognition and anomaly detection

## Customization Points

The system can be extended in these areas:

1. Add new data connectors for different markets (stocks, commodities, etc.)
2. Implement additional technical indicators and features
3. Create new market condition identifiers
4. Develop specialized analysis modules
5. Enhance AI integration with custom prompts and models

## Database and Storage

- Market data is cached in CSV files in the `cache/` directory
- OHLCV data is stored in `cache/ohlcv/`
- Technical indicators can be cached in `cache/indicators/`

## Search Guide

When looking for specific functionality:

1. **Data fetching/handling**: Look in `quant_system/data/` directory
2. **Technical indicators**: Check `quant_system/features/technical.py`
3. **Market condition logic**: See `quant_system/analysis/market_structure.py`
4. **Backtesting code**: Examine `quant_system/analysis/backtest.py`
5. **CLI commands**: Review `quant_system/interface/cli.py`
6. **API endpoints**: Explore `quant_system/interface/api.py`
7. **Logging and utilities**: Check `quant_system/utils.py`

## Common Workflows

### Adding a New Feature

1. Implement the feature calculation in `features/technical.py`
2. Add condition recognition in `analysis/market_structure.py`
3. Update the backtest functionality if needed
4. Expose through CLI/API as appropriate

### Adding a New Market

1. Create or extend data connector in `data/connectors.py`
2. Add specific technical indicators if needed
3. Implement market-specific analysis in the analysis module
4. Update CLI/API to support the new market

### Creating a New Command

1. Add the command parser in `interface/cli.py`
2. Implement the command handler method in `QuantSystemCLI` class
3. Register the command in the main function
4. Update documentation 