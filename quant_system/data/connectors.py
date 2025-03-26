import os
import pandas as pd
import ccxt
import traceback
from datetime import datetime, timedelta
import sys

# Add the project root to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our logging utilities
from quant_system.utils import get_logger, ErrorHandler

# Initialize logger for this module
logger = get_logger("data.connectors")

class CryptoDataConnector:
    """Connector for cryptocurrency market data"""
    
    def __init__(self, exchange_id='coinbase'):
        logger.info(f"Initializing CryptoDataConnector with exchange: {exchange_id}")
        try:
            self.exchange = getattr(ccxt, exchange_id)({
                'enableRateLimit': True,
            })
            logger.debug(f"Successfully connected to {exchange_id} API")
        except Exception as e:
            logger.error(f"Failed to initialize exchange {exchange_id}: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def fetch_ohlcv(self, symbol='BTC/USD', timeframe='1d', limit=100):
        """Fetch OHLCV data for a given symbol"""
        logger.info(f"Fetching OHLCV data for {symbol} ({timeframe}, limit={limit})")
        try:
            logger.debug(f"Sending API request to fetch {symbol} OHLCV data")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No data returned for {symbol} ({timeframe})")
                return pd.DataFrame()
            
            # Create dataframe from response
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Log successful fetch
            first_date = df.index[0].strftime('%Y-%m-%d')
            last_date = df.index[-1].strftime('%Y-%m-%d')
            logger.info(f"Successfully fetched {len(df)} records for {symbol} from {first_date} to {last_date}")
            
            return df
        except ccxt.NetworkError as e:
            logger.error(f"Network error while fetching {symbol} data: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
        except ccxt.InvalidSymbol:
            logger.error(f"Invalid symbol: {symbol}")
            return pd.DataFrame()
        except ccxt.RequestTimeout:
            logger.error(f"Request timeout while fetching {symbol} data")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def fetch_market_data(self, symbols=['BTC/USD', 'ETH/USD'], timeframe='1d', days=30):
        """Fetch market data for multiple symbols"""
        logger.info(f"Fetching market data for {len(symbols)} symbols ({timeframe}, {days} days)")
        
        result = {}
        with ErrorHandler(context="fetching multiple symbols data") as handler:
            for symbol in symbols:
                logger.debug(f"Processing symbol: {symbol}")
                df = self.fetch_ohlcv(symbol, timeframe, limit=days)
                if not df.empty:
                    result[symbol] = df
                else:
                    logger.warning(f"No data available for {symbol}, skipping")
        
        logger.info(f"Successfully fetched data for {len(result)}/{len(symbols)} symbols")
        return result
        
    def fetch_latest_ticker(self, symbol='BTC/USD'):
        """Fetch the latest ticker information for a symbol"""
        logger.info(f"Fetching latest ticker for {symbol}")
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.debug(f"Ticker data received for {symbol}")
            
            # Log the important metrics
            logger.info(f"{symbol} last price: {ticker['last']}, 24h change: {ticker.get('percentage', 0):.2f}%")
            
            return ticker
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return None
            
    def fetch_order_book(self, symbol='BTC/USD', limit=20):
        """Fetch the current order book for a symbol"""
        logger.info(f"Fetching order book for {symbol} (limit={limit})")
        
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            
            # Log order book stats
            top_bid = order_book['bids'][0][0] if order_book['bids'] else None
            top_ask = order_book['asks'][0][0] if order_book['asks'] else None
            
            if top_bid and top_ask:
                spread = (top_ask - top_bid) / top_bid * 100
                logger.info(f"{symbol} order book - Top bid: {top_bid}, Top ask: {top_ask}, Spread: {spread:.3f}%")
            else:
                logger.warning(f"Incomplete order book data for {symbol}")
            
            return order_book
        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return None
