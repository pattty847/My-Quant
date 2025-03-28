# Quant System - Intelligent Market Analysis

This system provides advanced quantitative analysis for financial markets with AI-powered insights.

![kmeans_regimes_time_20250327_164930](https://github.com/user-attachments/assets/f25cf161-37bd-4ee2-bf1c-bb03cd1e4adc)
![pca_scatter_20250327_164920](https://github.com/user-attachments/assets/a5fdf14d-c6c4-4b29-ac43-5eb17bc6ede4)

## Features

- **Market Analysis**: Technical indicators, regime detection, and condition identification
- **AI-Powered Insights**: Natural language explanations of market conditions and patterns
- **Advanced Statistics**: PCA dimensionality reduction and Hidden Markov Models for regime detection
- **Backtesting**: Test specific market conditions against historical data
- **Data Management**: Smart caching system with complete historical data retrieval
- **Interactive Charts**: Visualize price action with AI explanations of what you're seeing

## Vision: AI-Powered Market Narrative

The core vision of this system is to provide traders with an **AI-powered explanation tool** that:

- Takes the current chart's OHLCV data and statistical measures
- Creates a narrative around price action and volume
- Explains potential smart money positioning
- Makes complex statistical analysis (like PCA and HMM) accessible through simple explanations
- Provides context for what you're seeing on the chart

This "explain tool" serves as your personal market analyst, giving you the story behind the numbers.

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

#### Advanced Analysis (PCA & Market Regimes)

```bash
# Run advanced statistical analysis with regime detection
python -m interface.cli advanced --symbol BTC/USDT --timeframe 1d --days 500 --components 2 --regimes 3
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

## Next Steps in Development

The system is being actively developed with these features coming soon:

1. **Interactive Chart UI**: Web-based interface with real-time data visualization
2. **AI Explanation Button**: One-click feature to explain chart patterns and market conditions
3. **Improved Visualization**: Interactive charts for better exploration of market regimes
4. **Enhanced AI Models**: More sophisticated narrative generation around price action and volume

## Troubleshooting

- **API connection issues**: Ensure you're using supported symbols and timeframes (use the `symbols` and `timeframes` commands to check)
- **LLM integration errors**: Verify your API key and provider settings in the .env file
- **Missing indicators**: Some indicators require sufficient historical data - try increasing the --days parameter
- **Error diagnosis**: Check the log files in the `logs/` directory for detailed error information and tracebacks

## Support

For issues or feature requests, please submit them to the project's issue tracker or contact the project maintainers.
