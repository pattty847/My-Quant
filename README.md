# Quant System - Installation & Quick Start Guide

This guide will help you set up and start using the Quant System for market analysis.

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
```

#### List available symbols

```bash
python -m interface.cli list-symbols
```

#### Backtest specific market conditions

```bash
python -m interface.cli backtest "RSI below 30" "MACD bullish crossover" --symbol ETH/USDT
```

### API Server

Start the API server:

```bash
# Using the main script
python main.py --mode api

# Or using the entry point (if installed with -e)
quant-api
```

The API documentation will be available at http://localhost:8000/docs

## Basic Usage Examples

### Command-Line

```bash
# Analyze Bitcoin on a daily timeframe
quant-system analyze --symbol BTC/USDT --timeframe 1d

# Analyze Ethereum on a 4-hour timeframe over the last 90 days
quant-system analyze --symbol ETH/USDT --timeframe 4h --days 90

# Backtest oversold conditions for Solana
quant-system backtest "Oversold (RSI below 30)" --symbol SOL/USDT --days 180
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

- **API connection issues**: Ensure you're using supported symbols and timeframes (use the list-symbols and list-timeframes commands to check)
- **LLM integration errors**: Verify your API key and provider settings in the .env file
- **Missing indicators**: Some indicators require sufficient historical data - try increasing the --days parameter

## Support

For issues or feature requests, please submit them to the project's issue tracker or contact the project maintainers.
