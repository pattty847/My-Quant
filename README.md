# Quant System - Installation & Quick Start Guide

This guide will help you set up and start using the Quant System for market analysis.

![kmeans_regimes_time_20250327_164930](https://github.com/user-attachments/assets/f25cf161-37bd-4ee2-bf1c-bb03cd1e4adc)
![pca_scatter_20250327_164920](https://github.com/user-attachments/assets/a5fdf14d-c6c4-4b29-ac43-5eb17bc6ede4)

## Prerequisites

- Python 3.8+
- pip (Python package installer)
- Git (optional)

## Installation

### 1. Clone or download the repository

```bash
git clone https://github.com/yourusername/quant-system.git
cd quant-system
```

Or download and extract the ZIP file.

### 2. Create and activate a virtual environment (recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install the package

```bash
# Install core dependencies
pip install -e .

# For development tools (optional)
pip install -e ".[dev]"

# For visualization tools (optional)
pip install -e ".[viz]"
```

### 4. Configure environment variables

```bash
# Copy the template file
cp .env.template .env

# Edit the .env file with your preferred text editor
# At minimum, you'll need to add an LLM API key if you want to use that feature
```

## Quick Start

### Command-Line Interface

#### Analyze a market

```bash
# Using the CLI directly
python -m interface.cli analyze --symbol BTC/USDT --timeframe 1d --days 365

# Or using the entry point (if installed with -e)
quant-system analyze --symbol BTC/USDT --timeframe 1d --days 365

# Enable debug logging with the --debug flag
quant-system analyze --symbol BTC/USDT --debug
```

#### List available symbols

```bash
python -m interface.cli symbols
```

#### Backtest specific market conditions

```bash
python -m interface.cli backtest --conditions "RSI below 30" "MACD bullish crossover" --symbol ETH/USDT
```

### API Server

Start the API server:

```bash
# Using the main script
python main.py api

# With debug logging enabled
python main.py api --debug

# Or using the entry point (if installed with -e)
quant-api
```

The API documentation will be available at http://localhost:8000/docs

## Logging System

The Quant System includes a comprehensive logging system that helps with debugging and monitoring:

- **Log Files**: All logs are saved in the `logs/` directory with timestamps
- **Debug Logging**: Use the `--debug` flag for detailed logging output
- **Error Tracebacks**: Errors are logged with full tracebacks for troubleshooting
- **Hierarchical Loggers**: Different components of the system use specific loggers

For more information, see [docs/logging.md](docs/logging.md).

## Basic Usage Examples

### Command-Line

```bash
# Analyze Bitcoin on a daily timeframe
quant-system analyze --symbol BTC/USDT --timeframe 1d

# Analyze Ethereum on a 4-hour timeframe over the last 90 days
quant-system analyze --symbol ETH/USDT --timeframe 4h --days 90

# Backtest oversold conditions for Solana with debug logging
quant-system backtest --conditions "Oversold (RSI below 30)" --symbol SOL/USDT --days 180 --debug
```

### API Examples

Once the API server is running, you can make requests:

```bash
# Using curl
curl -X POST "http://localhost:8000/analyze" -H "Content-Type: application/json" -d '{"symbol":"BTC/USDT", "timeframe":"1d", "days":90}'

# Get current market conditions
curl "http://localhost:8000/conditions?symbol=ETH/USDT"
```

You can also use the interactive Swagger documentation at http://localhost:8000/docs to test the API.

## Next Steps

1. **Add more data sources**: Implement connectors for traditional markets or macroeconomic data
2. **Enhance visualization**: Add plotly charts for better visualization of market conditions
3. **Implement advanced analytics**: Add pattern recognition or machine learning features
4. **Memory storage**: Add a database to store historical predictions and results

## Troubleshooting

- **API connection issues**: Ensure you're using supported symbols and timeframes (use the `symbols` and `timeframes` commands to check)
- **LLM integration errors**: Verify your API key and provider settings in the .env file
- **Missing indicators**: Some indicators require sufficient historical data - try increasing the --days parameter
- **Error diagnosis**: Check the log files in the `logs/` directory for detailed error information and tracebacks

## Support

For issues or feature requests, please submit them to the project's issue tracker or contact the project maintainers.
