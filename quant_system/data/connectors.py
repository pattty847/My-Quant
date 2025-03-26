# quant_system/data/connectors.py
import os
import pandas as pd
import ccxt
import traceback
from datetime import datetime, timedelta
import sys
import time

# Add the project root to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our logging utilities and cache
from quant_system.utils import get_logger, ErrorHandler
from quant_system.data.cache import DataCache

# Initialize logger for this module
logger = get_logger("data.connectors")

class CryptoDataConnector:
    """Connector for cryptocurrency market data with caching"""
    
    def __init__(self, exchange_id='coinbase', use_cache=True, cache_dir="cache"):
        """Initialize the data connector
        
        Args:
            exchange_id: ID of the exchange to use (e.g., 'coinbase', 'binance')
            use_cache: Whether to use data caching
            cache_dir: Directory for cached data
        """
        logger.info(f"Initializing CryptoDataConnector with exchange: {exchange_id}")
        self.use_cache = use_cache
        
        try:
            self.exchange = getattr(ccxt, exchange_id)({
                'enableRateLimit': True,
            })
            logger.debug(f"Successfully connected to {exchange_id} API")
            
            # Initialize cache if enabled
            if self.use_cache:
                self.cache = DataCache(cache_dir=cache_dir)
                logger.info("Data caching enabled")
            else:
                self.cache = None
                logger.info("Data caching disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize exchange {exchange_id}: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def fetch_paginated_ohlcv(self, symbol='BTC/USD', timeframe='1d', 
                              from_datetime=None, to_datetime=None, retry_delay=30):
        """Fetch complete OHLCV data for a given period using pagination
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '1d', '1h')
            from_datetime: Start datetime as string 'YYYY-MM-DD HH:MM:SS' or datetime object
            to_datetime: End datetime as string 'YYYY-MM-DD HH:MM:SS' or datetime object
            retry_delay: Delay in seconds before retrying on error
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching paginated OHLCV data for {symbol} ({timeframe}) from {from_datetime} to {to_datetime}")
        
        # Convert string datetimes to timestamp if provided
        if from_datetime is not None:
            if isinstance(from_datetime, str):
                # Add time component if missing
                if len(from_datetime) == 10:  # YYYY-MM-DD format
                    from_datetime = f"{from_datetime} 00:00:00"
                from_timestamp = self.exchange.parse8601(from_datetime)
                if from_timestamp is None:
                    logger.error(f"Could not parse from_datetime: {from_datetime}")
                    from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
            elif isinstance(from_datetime, datetime):
                from_timestamp = int(from_datetime.timestamp() * 1000)
            else:
                from_timestamp = from_datetime  # Assume it's already a timestamp
        else:
            # Default to 1 year ago if not specified
            from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
            
        # Set end timestamp
        if to_datetime is not None:
            if isinstance(to_datetime, str):
                # Add time component if missing
                if len(to_datetime) == 10:  # YYYY-MM-DD format
                    to_datetime = f"{to_datetime} 23:59:59"
                to_timestamp = self.exchange.parse8601(to_datetime)
                if to_timestamp is None:
                    logger.error(f"Could not parse to_datetime: {to_datetime}")
                    to_timestamp = int(datetime.now().timestamp() * 1000)
            elif isinstance(to_datetime, datetime):
                to_timestamp = int(to_datetime.timestamp() * 1000)
            else:
                to_timestamp = to_datetime  # Assume it's already a timestamp
        else:
            to_timestamp = int(datetime.now().timestamp() * 1000)
            
        logger.debug(f"Timestamp range: {from_timestamp} to {to_timestamp}")
        
        # First, check if we already have this data in the standard cache
        if self.use_cache:
            cached_data = self.cache.get_cached_ohlcv(symbol, timeframe, max_age_days=None)
            
            if cached_data is not None and not cached_data.empty:
                logger.info(f"Found existing cached data for {symbol} ({len(cached_data)} records)")
                
                # Filter to the requested time range
                from_dt = datetime.fromtimestamp(from_timestamp / 1000)
                to_dt = datetime.fromtimestamp(to_timestamp / 1000)
                filtered_data = cached_data[
                    (cached_data.index >= from_dt) & 
                    (cached_data.index <= to_dt)
                ]
                
                # Check if we have complete data for the requested range
                if len(filtered_data) > 0:
                    has_start = filtered_data.index.min() <= from_dt
                    has_end = filtered_data.index.max() >= to_dt
                    
                    if has_start and has_end:
                        logger.info(f"Complete data found in cache for requested time range ({len(filtered_data)} records)")
                        return filtered_data
                    
                    # If we have partial data, we could fetch just what's missing
                    # But for simplicity, we'll fetch everything and merge with existing cache
            
        # Fetch data with pagination
        all_ohlcv = []
        current_timestamp = from_timestamp
        
        while current_timestamp < to_timestamp:
            try:
                logger.debug(f"Fetching candles from {self.exchange.iso8601(current_timestamp)}")
                ohlcvs = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp)
                
                if not ohlcvs or len(ohlcvs) == 0:
                    logger.warning(f"No data returned at timestamp {current_timestamp}, stopping pagination")
                    break
                    
                logger.debug(f"Fetched {len(ohlcvs)} candles")
                
                # Filter out any data beyond our end timestamp
                ohlcvs = [candle for candle in ohlcvs if candle[0] <= to_timestamp]
                
                # Add to our result set
                all_ohlcv.extend(ohlcvs)
                
                # Update timestamp for next iteration based on last candle
                if len(ohlcvs) > 0:
                    # Move timestamp forward by 1ms to avoid duplicates
                    current_timestamp = ohlcvs[-1][0] + 1
                else:
                    # If we got an empty response but didn't break earlier, move forward in time
                    timeframe_ms = self._timeframe_to_milliseconds(timeframe)
                    current_timestamp += timeframe_ms * 100  # Skip ahead 100 candles
                
                # If we got fewer candles than expected, we might be at the end
                if len(ohlcvs) < 100:  # Most exchanges return max 100-1000 candles per request
                    logger.debug("Received fewer candles than expected, may have reached the end")
                    # Only break if we actually got some candles, otherwise continue
                    if len(ohlcvs) > 0:
                        break
            
            except (ccxt.ExchangeError, ccxt.AuthenticationError,
                   ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
                error_msg = f"Error fetching data: {type(error).__name__}, {error.args}"
                logger.error(error_msg)
                logger.info(f"Retrying in {retry_delay} seconds...")
                
                # Wait before retrying
                time.sleep(retry_delay)
            
            except Exception as e:
                logger.error(f"Unexpected error fetching paginated data: {e}")
                logger.debug(traceback.format_exc())
                break
        
        # Convert to DataFrame
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates that might occur at pagination boundaries
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            
            logger.info(f"Successfully fetched {len(df)} records from {df.index[0]} to {df.index[-1]}")
            
            # Update the standard OHLCV cache
            if self.use_cache:
                logger.debug(f"Updating standard OHLCV cache for {symbol}")
                combined_df = self.cache.update_ohlcv_cache(symbol, timeframe, df)
                
                # Return only the requested time range
                from_dt = datetime.fromtimestamp(from_timestamp / 1000)
                to_dt = datetime.fromtimestamp(to_timestamp / 1000)
                result_df = combined_df[
                    (combined_df.index >= from_dt) & 
                    (combined_df.index <= to_dt)
                ]
                return result_df
            
            return df
        else:
            logger.warning(f"No data fetched for {symbol} in the specified time range")
            return pd.DataFrame()
    
    def _timeframe_to_milliseconds(self, timeframe):
        """Convert a timeframe string to milliseconds"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60 * 1000
        else:
            logger.warning(f"Unknown timeframe unit: {unit}, defaulting to days")
            return value * 24 * 60 * 60 * 1000
    
    def fetch_ohlcv(self, symbol='BTC/USD', timeframe='1d', limit=100, force_refresh=False):
        """Fetch OHLCV data for a given symbol, with caching
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '1d', '1h')
            limit: Number of candles to fetch
            force_refresh: Whether to ignore cache and fetch new data
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching OHLCV data for {symbol} ({timeframe}, limit={limit})")
        
        # Try to get data from cache if enabled and not forcing refresh
        if self.use_cache and not force_refresh:
            cached_data = self.cache.get_cached_ohlcv(symbol, timeframe)
            
            if cached_data is not None and len(cached_data) >= limit:
                logger.info(f"Using cached data for {symbol} ({len(cached_data)} records)")
                
                # Check if we need to update the latest candle
                latest_candle = cached_data.iloc[-1]
                latest_candle_time = latest_candle.name
                current_time = pd.Timestamp.now()
                
                # Calculate the timeframe in minutes
                timeframe_minutes = self._timeframe_to_minutes(timeframe)
                
                # If the latest candle is from the current period, we should update it
                if (current_time - latest_candle_time).total_seconds() < timeframe_minutes * 60:
                    logger.info(f"Latest candle is from current period, updating...")
                    # Fetch just the latest candle to update
                    new_data = self._fetch_from_api(symbol, timeframe, limit=1)
                    if not new_data.empty:
                        # Update the latest candle in cache
                        cached_data.iloc[-1] = new_data.iloc[-1]
                        self.cache.cache_ohlcv(symbol, timeframe, cached_data)
                
                return cached_data.iloc[-limit:]
            
            elif cached_data is not None:
                logger.info(f"Cached data insufficient ({len(cached_data)} < {limit} records), fetching additional data")
                
                # Calculate how many additional records we need
                # Add some buffer to account for potential gaps
                additional_limit = limit - len(cached_data) + 10
                
                # Fetch additional data and merge with cache
                new_data = self._fetch_from_api(symbol, timeframe, additional_limit)
                
                if not new_data.empty:
                    # Update cache with new data
                    combined_data = self.cache.update_ohlcv_cache(symbol, timeframe, new_data)
                    logger.info(f"Combined data has {len(combined_data)} records")
                    
                    # Return only the requested number of records
                    return combined_data.iloc[-limit:]
                else:
                    # If API fetch failed, return whatever we have in cache
                    logger.warning(f"Failed to fetch additional data for {symbol}, using cached data only")
                    return cached_data.iloc[-limit:]
        
        # Fetch data from API if cache is disabled or we need a refresh
        data = self._fetch_from_api(symbol, timeframe, limit)
        
        # Cache the data if enabled
        if self.use_cache and not data.empty:
            # If we're forcing a refresh, update rather than overwrite
            if force_refresh:
                data = self.cache.update_ohlcv_cache(symbol, timeframe, data)
            else:
                self.cache.cache_ohlcv(symbol, timeframe, data)
        
        return data
    
    def _timeframe_to_minutes(self, timeframe):
        """Convert a timeframe string to minutes"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 24 * 60
        elif unit == 'w':
            return value * 7 * 24 * 60
        else:
            logger.warning(f"Unknown timeframe unit: {unit}, defaulting to days")
            return value * 24 * 60
    
    def _fetch_from_api(self, symbol, timeframe, limit):
        """Fetch OHLCV data directly from the exchange API
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.debug(f"Fetching {symbol} from API ({timeframe}, limit={limit})")
        try:
            # Increase the limit slightly to account for potential missing data
            api_limit = min(limit + 10, 1000)  # Most exchanges have a 1000 limit
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=api_limit)
            
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No data returned from API for {symbol} ({timeframe})")
                return pd.DataFrame()
            
            # Create dataframe from response
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Log successful fetch
            first_date = df.index[0].strftime('%Y-%m-%d')
            last_date = df.index[-1].strftime('%Y-%m-%d')
            logger.info(f"Successfully fetched {len(df)} records for {symbol} from API ({first_date} to {last_date})")
            
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
    
    def fetch_extended_history(self, symbol='BTC/USD', timeframe='1d', days=365):
        """Fetch extended historical data, potentially making multiple API calls
        
        This method is useful for getting long-term historical data beyond
        the exchange's single request limit.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with historical OHLCV data
        """
        logger.info(f"Fetching extended history for {symbol} ({timeframe}, {days} days)")
        
        # Try to get from cache first
        if self.use_cache:
            cached_data = self.cache.get_cached_ohlcv(symbol, timeframe)
            
            if cached_data is not None and len(cached_data) >= days:
                logger.info(f"Using cached extended history for {symbol}")
                # Return the requested number of days
                return cached_data.iloc[-days:]
        
        # Convert days to the number of candles needed based on timeframe
        candles_per_day = {
            '1m': 1440, '5m': 288, '15m': 96, '30m': 48, 
            '1h': 24, '4h': 6, '6h': 4, '12h': 2, '1d': 1
        }
        
        # Default to 1 if unknown timeframe
        candles_per_day_value = candles_per_day.get(timeframe, 1)
        total_candles = days * candles_per_day_value
        
        # Most exchanges limit to 1000 candles per request
        max_per_request = 1000
        
        if total_candles <= max_per_request:
            # We can fetch all data in one request
            logger.debug(f"Fetching {total_candles} candles in a single request")
            df = self._fetch_from_api(symbol, timeframe, total_candles)
            
            # Cache result
            if self.use_cache and not df.empty:
                self.cache.cache_ohlcv(symbol, timeframe, df)
                
            return df
            
        else:
            # Need multiple requests - start with any cached data
            logger.debug(f"Need multiple requests for {total_candles} candles")
            result_df = pd.DataFrame()
            
            if self.use_cache:
                cached_data = self.cache.get_cached_ohlcv(symbol, timeframe, max_age_days=None)
                if cached_data is not None and not cached_data.empty:
                    result_df = cached_data.copy()
                    logger.info(f"Starting with {len(result_df)} cached records")
            
            # Calculate how many more candles we need
            candles_needed = total_candles - len(result_df)
            
            if candles_needed <= 0:
                logger.info(f"No additional candles needed, using cached data")
                return result_df.iloc[-total_candles:]
            
            # Make multiple requests as needed
            remaining = candles_needed
            max_attempts = 10  # Limit the number of requests to avoid rate limits
            attempts = 0
            
            while remaining > 0 and attempts < max_attempts:
                attempts += 1
                batch_size = min(remaining, max_per_request)
                
                logger.info(f"Fetching batch {attempts}: {batch_size} candles")
                
                # If we have data, use the oldest timestamp to fetch older data
                if not result_df.empty:
                    # Get earliest timestamp and convert to milliseconds for the 'since' parameter
                    earliest_ts = int(result_df.index[0].timestamp() * 1000)
                    new_batch = self._fetch_from_api_with_since(symbol, timeframe, batch_size, since=earliest_ts - 1)
                else:
                    new_batch = self._fetch_from_api(symbol, timeframe, batch_size)
                
                if new_batch.empty:
                    logger.warning(f"Failed to fetch batch {attempts}, stopping")
                    break
                
                # Combine with existing data
                result_df = pd.concat([new_batch, result_df])
                result_df = result_df[~result_df.index.duplicated(keep='first')]
                result_df = result_df.sort_index()
                
                logger.info(f"Combined data now has {len(result_df)} records")
                
                # Update remaining count
                remaining = total_candles - len(result_df)
                
                # If we didn't get as many candles as requested, we've likely reached the limit
                if len(new_batch) < batch_size:
                    logger.info(f"Received fewer candles than requested ({len(new_batch)} < {batch_size}), assuming complete")
                    break
            
            # Cache the final result
            if self.use_cache and not result_df.empty:
                self.cache.cache_ohlcv(symbol, timeframe, result_df)
                
            # Return only the requested number of days
            return result_df.iloc[-total_candles:] if len(result_df) > total_candles else result_df
    
    def _fetch_from_api_with_since(self, symbol, timeframe, limit, since=None):
        """Fetch OHLCV data from a specific timestamp
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            limit: Number of candles to fetch
            since: Timestamp (in milliseconds) to fetch from
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.debug(f"Fetching {symbol} from API with since={since}")
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No data returned from API for {symbol} with since={since}")
                return pd.DataFrame()
            
            # Create dataframe from response
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Log successful fetch
            first_date = df.index[0].strftime('%Y-%m-%d')
            last_date = df.index[-1].strftime('%Y-%m-%d')
            logger.info(f"Successfully fetched {len(df)} records for {symbol} from API ({first_date} to {last_date})")
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} with since={since}: {e}")
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