# Code Reference Guide

This document provides reference examples of key functions and code patterns used throughout the project.

## Data Fetching

### Unified Market Data Fetching

The system can fetch complete historical data using the unified `fetch_market_data` function, allowing for retrieval of large datasets that exceed the exchange's single request limit.

```python
# Example: Fetching complete BTC/USD history from 2019 to present
from quant_system.data.connectors import CryptoDataConnector

data_connector = CryptoDataConnector(use_cache=True)
historical_data = data_connector.fetch_market_data(
    symbol="BTC/USD",
    timeframe="1d",
    from_date="2019-01-01",
    to_date="2023-01-01",
    add_indicators=True
)
```

## CLI Commands

### Analyzing a Market

```bash
# Basic analysis
python main.py cli analyze --symbol BTC/USDT

# Advanced analysis with timeframe and period settings
python main.py cli analyze --symbol ETH/USDT --timeframe 1h --days 90 --debug
```

### Backtesting Market Conditions

```bash
# Test a single condition
python main.py cli backtest --conditions "Golden cross" --symbol BTC/USDT

# Test multiple conditions
python main.py cli backtest --conditions "Oversold RSI" "Bullish trend" --symbol ETH/USDT --timeframe 4h
```

### Fetching Historical Data

```bash
# Fetch historical data for a specific period
python main.py cli history --symbol BTC/USDT --from 2020-01-01 --to 2023-01-01

# Fetch data and save to CSV
python main.py cli history --symbol ETH/USDT --from 2018-01-01 --output data/eth_history.csv
```

## Technical Analysis

### Adding Indicators to Price Data

```python
from quant_system.features.technical import TechnicalFeatures

tech = TechnicalFeatures()
df_with_indicators = tech.add_indicators(price_data)

# Access calculated indicators
rsi = df_with_indicators['rsi']
sma_50 = df_with_indicators['sma_50']
```

### Identifying Market Conditions

```python
from quant_system.analysis.market_structure import MarketStructureAnalyzer

analyzer = MarketStructureAnalyzer()
conditions = analyzer.identify_conditions(df_with_indicators)

print(f"Current market conditions: {', '.join(conditions)}")
```

## Backtesting

### Finding Similar Historical Conditions

```python
from quant_system.analysis.backtest import MarketBacktester

backtester = MarketBacktester()
similar_dates = backtester.find_similar_conditions(
    df_with_indicators, 
    conditions=["Oversold RSI", "Volume spike"]
)
```

### Analyzing Forward Returns

```python
results_df, stats = backtester.analyze_forward_returns(df_with_indicators, similar_dates)

# Print average returns for different periods
for period, value in stats.items():
    if 'mean' in period:
        days = period.split('d_')[0]
        print(f"{days}-day forward return (avg): {value:.2f}%")
```

## Caching

### Managing the Data Cache

```python
from quant_system.data.cache import DataCache

# Initialize cache
cache = DataCache(cache_dir="cache")

# Get cached data
cached_data = cache.get_cached_ohlcv("BTC/USDT", "1d")

# Update cache with new data
cache.update_ohlcv_cache("BTC/USDT", "1d", new_data)

# Clear cache for a specific symbol
cache.clear_cache(symbol="BTC/USDT")

# Get cache statistics
stats = cache.get_cache_stats()
print(f"Cache size: {stats['total_size_mb']} MB")
print(f"Hit ratio: {stats['hit_ratio']}")
```

## Error Handling

The project uses a context manager pattern for standardized error handling:

```python
from quant_system.utils import ErrorHandler

with ErrorHandler(context="fetching market data") as handler:
    # Code that might raise exceptions
    data = api.fetch_data()
    results = process_data(data)
    # If an exception occurs, it will be logged with context
```

## Logging

```python
from quant_system.utils import get_logger, DEBUG, INFO

# Get a logger for a specific module
logger = get_logger("my_module")

# Log at different levels
logger.debug("Detailed information for debugging")
logger.info("Confirmation that things are working")
logger.warning("Something unexpected happened")
logger.error("A more serious problem occurred")
logger.critical("Critical error causing program failure")

# Set log level
from quant_system.utils import set_log_level
set_log_level(DEBUG)  # Show all logs including debug
```

## LLM Integration

```python
from quant_system.analysis.llm_interface import LLMAnalyzer

llm = LLMAnalyzer(api_key="your_api_key")
market_summary = llm.generate_market_summary(
    df_indicators, 
    conditions=["Bullish trend", "Support test"],
    stats=performance_stats
)
```

## API Usage

```python
# Start the API server
python main.py api --port 8000

# Make requests to the API
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC/USDT", "timeframe": "1d", "days": 100}'
``` 