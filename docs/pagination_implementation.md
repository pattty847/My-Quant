# OHLCV Pagination Implementation Guide

This document explains the implementation of paginated OHLCV (Open-High-Low-Close-Volume) data fetching in the quantitative analysis system.

## Overview

Many cryptocurrency exchanges and financial data APIs limit the amount of historical data you can retrieve in a single request (often 100-1000 candles). The pagination implementation allows fetching complete historical data by making multiple sequential requests and combining the results.

## Implementation Details

The implementation is in the `fetch_paginated_ohlcv` method of the `CryptoDataConnector` class in `quant_system/data/connectors.py`.

### Key Features

1. **Date Range Support**: Fetch data between specific start and end dates
2. **Smart Caching**: Uses the existing cache system to minimize API calls
3. **Error Handling**: Graceful handling of API errors with retries
4. **Timestamp Management**: Properly handles pagination boundaries to avoid duplicates
5. **Data Filtering**: Ensures only data within the requested time range is returned

### How It Works

The pagination algorithm follows these steps:

1. Convert input date strings to timestamps
2. Check if requested data exists in cache
3. If complete data is in cache, return it directly
4. Otherwise, fetch data in chunks through pagination:
   - Start with the earliest timestamp
   - Fetch a batch of candles
   - Add to the result set
   - Use the timestamp of the last candle to determine the next batch
   - Repeat until all data is fetched
5. Remove duplicates and sort by timestamp
6. Update the cache with the new data
7. Return the filtered result set

### Code Example

```python
def fetch_paginated_ohlcv(self, symbol='BTC/USD', timeframe='1d', 
                         from_datetime=None, to_datetime=None, retry_delay=30):
    """Fetch complete OHLCV data for a given period using pagination"""
    
    # Convert dates to timestamps
    from_timestamp = self._convert_to_timestamp(from_datetime)
    to_timestamp = self._convert_to_timestamp(to_datetime, default_to_now=True)
    
    # Check cache first
    if self.use_cache:
        cached_data = self.cache.get_cached_ohlcv(symbol, timeframe, max_age_days=None)
        if cached_data is not None and not cached_data.empty:
            # Filter to the requested time range
            # Return if complete data exists in cache
            pass
    
    # Fetch data with pagination
    all_ohlcv = []
    current_timestamp = from_timestamp
    
    while current_timestamp < to_timestamp:
        try:
            # Fetch a batch of candles
            ohlcvs = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp)
            
            # Add to result set
            all_ohlcv.extend(ohlcvs)
            
            # Update timestamp for next iteration
            if len(ohlcvs) > 0:
                current_timestamp = ohlcvs[-1][0] + 1
            else:
                # Handle empty response
                break
                
        except Exception as e:
            # Handle errors and retry
            pass
    
    # Convert to DataFrame
    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        
        # Update cache
        if self.use_cache:
            self.cache.update_ohlcv_cache(symbol, timeframe, df)
        
        return df
    else:
        return pd.DataFrame()
```

## Usage in CLI

The pagination functionality is exposed through the CLI with the `history` command:

```bash
python main.py cli history --symbol BTC/USD --from 2020-01-01 --to 2023-01-01
```

The implementation is in the `fetch_history` method of the `QuantSystemCLI` class in `quant_system/interface/cli.py`.

## Usage in Code

```python
from quant_system.data.connectors import CryptoDataConnector

# Initialize connector
connector = CryptoDataConnector(use_cache=True)

# Fetch complete history since 2018
data = connector.fetch_paginated_ohlcv(
    symbol="BTC/USD",
    timeframe="1d",
    from_datetime="2018-01-01",
    to_datetime=None  # Defaults to current time
)

# Process or analyze the data
print(f"Retrieved {len(data)} candles from {data.index[0]} to {data.index[-1]}")
```

## How Pagination Handles Exchange Limits

Different exchanges have different limits:

- Coinbase Pro: 300 candles per request
- Binance: 1000 candles per request
- Kraken: 720 candles per request

The pagination implementation works with these limits by:

1. Requesting a batch of candles up to the exchange's limit
2. Using the timestamp of the last candle + 1ms as the starting point for the next request
3. Continuing until we reach the end date
4. Handling rate limiting automatically through CCXT's built-in rate limiter

## Edge Cases Handled

The implementation handles various edge cases:

1. **Empty responses**: If an exchange returns no data for a time range
2. **Duplicates**: Candles at pagination boundaries might be returned twice
3. **Rate limiting**: Respects exchange rate limits to avoid being blocked
4. **Network errors**: Retries on temporary connection issues
5. **Partial data**: Deals with having some but not all data in cache

## Caching Considerations

The implementation integrates with the existing cache system:

1. It checks for cached data first to avoid unnecessary API calls
2. It updates the cache with new data
3. If partial data is in cache, it fetches only what's missing
4. The cache is used to fulfill future requests without API calls

## Future Improvements

Potential enhancements to consider:

1. Parallel fetching for multiple symbols
2. More granular cache updates for specific date ranges
3. Progressive display for long-running fetches
4. Adaptive retry mechanisms for unreliable connections
5. Scheduled background updates of frequently used datasets 