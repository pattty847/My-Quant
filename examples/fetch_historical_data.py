#!/usr/bin/env python
# examples/fetch_historical_data.py

"""
Example script showing how to fetch complete historical data using pagination.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our system components
from quant_system.data.connectors import CryptoDataConnector
from quant_system.utils import get_logger, set_log_level, DEBUG

# Initialize logger
logger = get_logger("examples.fetch_historical")
set_log_level(DEBUG)  # Set to DEBUG for verbose output

def main():
    """Example of fetching complete historical data"""
    print("Fetching Complete Historical Data Example")
    print("----------------------------------------\n")
    
    # Configuration
    symbol = "BTC/USD"
    timeframe = "1d"
    from_date = "2020-01-01"  # Start date
    # Default to current date for end date
    to_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Period: {from_date} to {to_date}\n")
    
    # Initialize data connector with caching enabled
    data_connector = CryptoDataConnector(use_cache=True)
    
    # Fetch complete historical data with pagination
    data = data_connector.fetch_paginated_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        from_datetime=from_date,
        to_datetime=to_date
    )
    
    if data.empty:
        print("Error: No data returned")
        return 1
    
    # Print statistics
    print(f"Successfully retrieved {len(data)} candles")
    print(f"Time range: {data.index[0]} to {data.index[-1]}\n")
    
    # Display sample data
    print("Data sample (first 5 rows):")
    pd.set_option('display.precision', 2)
    print(data.head(5).to_string())
    print("\nData sample (last 5 rows):")
    print(data.tail(5).to_string())
    
    # Calculate some basic statistics
    returns = data['close'].pct_change() * 100
    print("\nBasic Statistics:")
    print(f"Average Daily Return: {returns.mean():.2f}%")
    print(f"Daily Volatility: {returns.std():.2f}%")
    print(f"Max Daily Gain: {returns.max():.2f}%")
    print(f"Max Daily Loss: {returns.min():.2f}%")
    print(f"Positive Days: {(returns > 0).sum()} ({(returns > 0).mean() * 100:.1f}%)")
    
    # Optional: Save to CSV (comment this out if you don't need a separate file)
    # output_file = "historical_data.csv"
    # data.to_csv(output_file)
    # print(f"\nData saved to {output_file}")

    print("\nData is cached in the standard OHLCV cache for future use")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 