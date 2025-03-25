import os
import pandas as pd
import ccxt
from datetime import datetime, timedelta

class CryptoDataConnector:
    """Connector for cryptocurrency market data"""
    
    def __init__(self, exchange_id='coinbase'):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })
    
    def fetch_ohlcv(self, symbol='BTC/USD', timeframe='1d', limit=100):
        """Fetch OHLCV data for a given symbol"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def fetch_market_data(self, symbols=['BTC/USD', 'ETH/USD'], timeframe='1d', days=30):
        """Fetch market data for multiple symbols"""
        result = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, limit=days)
            if not df.empty:
                result[symbol] = df
        return result
