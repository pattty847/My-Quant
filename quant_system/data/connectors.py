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
    
    def fetch_extended_history(self, symbol='BTC/USD', timeframe='1d', days=365):
        """
        DEPRECATED: Use fetch_market_data instead.
        This method remains for backward compatibility only.
        """
        logger.warning("fetch_extended_history is deprecated, use fetch_market_data instead")
        
        # Calculate a date range going back 'days' from today
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Use the unified function
        return self.fetch_market_data(
            symbol=symbol, 
            timeframe=timeframe, 
            from_date=from_date,
            to_date=to_date
        )
    
    def fetch_market_data(self, symbol='BTC/USD', timeframe='1d', limit=None, 
                         from_date=None, to_date=None, add_indicators=False,
                         force_refresh=False, csv_output=None, retry_delay=30):
        """Unified market data fetching function that handles caching, pagination and indicators
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            limit: Optional number of candles to return (most recent)
            from_date: Optional start date as string 'YYYY-MM-DD' or datetime object
            to_date: Optional end date as string 'YYYY-MM-DD' or datetime object
            add_indicators: Whether to calculate and cache technical indicators
            force_refresh: Whether to force fetch from API ignoring cache
            csv_output: Optional path to save data as CSV
            retry_delay: Delay in seconds before retrying on error
            
        Returns:
            DataFrame with OHLCV data and optional indicators
        """
        market_data = None
        
        # 1. Determine what we need to fetch
        logger.info(f"Fetching market data for {symbol} ({timeframe})")
        
        # Convert date strings to timestamps if provided
        from_timestamp = None
        to_timestamp = None
        
        if from_date is not None:
            if isinstance(from_date, str):
                # Add time component if missing
                if len(from_date) == 10:  # YYYY-MM-DD format
                    from_date = f"{from_date} 00:00:00"
                from_timestamp = self.exchange.parse8601(from_date)
                if from_timestamp is None:
                    logger.error(f"Could not parse from_date: {from_date}")
                    # Default to 1 year ago if parsing fails
                    from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
            elif isinstance(from_date, datetime):
                # Default to 1 year ago if parsing fails
                from_timestamp = int(from_date.timestamp() * 1000)
            else:
                from_timestamp = from_date  # Assume it's already a timestamp
                
        if to_date is not None:
            if isinstance(to_date, str):
                # Add time component if missing
                if len(to_date) == 10:  # YYYY-MM-DD format
                    to_date = f"{to_date} 23:59:59"
                to_timestamp = self.exchange.parse8601(to_date)
                if to_timestamp is None:
                    logger.error(f"Could not parse to_date: {to_date}")
                    # Default to current timestamp if parsing fails
                    to_timestamp = int(datetime.now().timestamp() * 1000)
            elif isinstance(to_date, datetime):
                to_timestamp = int(to_date.timestamp() * 1000)
            else:
                to_timestamp = to_date  # Assume it's already a timestamp
        
        # 2. Check cache if enabled and not forcing refresh
        if self.use_cache and not force_refresh:
            cached_data = self.cache.get_cached_ohlcv(symbol, timeframe, max_age_days=1)
            
            if cached_data is not None and not cached_data.empty:
                logger.info(f"Found cached data for {symbol} ({len(cached_data)} records)")
                
                # If we have date range, filter the cached data
                if from_timestamp or to_timestamp:
                    if from_timestamp:
                        from_dt = datetime.fromtimestamp(from_timestamp / 1000)
                        cached_data = cached_data[cached_data.index >= from_dt]
                    if to_timestamp:
                        to_dt = datetime.fromtimestamp(to_timestamp / 1000)
                        cached_data = cached_data[cached_data.index <= to_dt]
                    
                    logger.debug(f"Filtered cached data to {len(cached_data)} records")
                
                # If we have limit, return only that many recent records
                if limit and len(cached_data) >= limit:
                    market_data = cached_data.iloc[-limit:]
                    logger.info(f"Using {len(market_data)} records from cache (limit={limit})")
                else:
                    # We have complete cached data for the request
                    market_data = cached_data
                
                # Check if we need to update with the latest data
                if market_data is not None:
                    # Check if the latest candle is from the current period
                    latest_candle_time = market_data.index[-1]
                    current_time = datetime.now()
                    
                    # Calculate the timeframe in minutes
                    timeframe_minutes = self._timeframe_to_minutes(timeframe)
                    
                    seconds_since_last_candle = (current_time - latest_candle_time).total_seconds()
                    
                    # If the latest candle is older than one candle period but less than two periods,
                    # we might need a new candle
                    if timeframe_minutes * 60 < seconds_since_last_candle < timeframe_minutes * 60 * 2:
                        logger.info(f"Checking for newer data since {latest_candle_time}")
                        
                        try:
                            new_candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=2)
                            if new_candles and len(new_candles) > 0:
                                newest_timestamp = new_candles[-1][0]
                                newest_dt = datetime.fromtimestamp(newest_timestamp / 1000)
                                
                                if newest_dt > latest_candle_time:
                                    logger.info(f"Found newer data, updating cache with candle from {newest_dt}")
                                    new_df = pd.DataFrame(new_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
                                    new_df.set_index('timestamp', inplace=True)
                                    
                                    # Add to cache
                                    updated_cache = pd.concat([cached_data, new_df])
                                    updated_cache = updated_cache[~updated_cache.index.duplicated(keep='last')]
                                    updated_cache = updated_cache.sort_index()
                                    
                                    # Update the cache
                                    self.cache.cache_ohlcv(symbol, timeframe, updated_cache)
                                    
                                    # Update our result
                                    if limit and len(updated_cache) >= limit:
                                        market_data = updated_cache.iloc[-limit:]
                                    else:
                                        market_data = updated_cache
                        except Exception as e:
                            logger.warning(f"Error checking for newer data: {e}")
        
        # 3. If we don't have complete data from cache, fetch from exchange
        if market_data is None or (
            (from_timestamp and market_data.index.min() > datetime.fromtimestamp(from_timestamp / 1000)) or
            (to_timestamp and market_data.index.max() < datetime.fromtimestamp(to_timestamp / 1000))
        ):
            logger.info("Need to fetch data from exchange")
            all_ohlcv = []
            
            # Determine if we need pagination
            need_pagination = from_timestamp is not None or limit is None or limit > 1000
            
            if need_pagination:
                # Use pagination to fetch all data
                logger.info("Using pagination to fetch complete data")
                
                # Start from the earliest timestamp we need
                current_timestamp = from_timestamp
                if current_timestamp is None:
                    # Default to 1 year ago if not specified
                    current_timestamp = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
                
                # End at the latest timestamp we need
                end_timestamp = to_timestamp
                if end_timestamp is None:
                    end_timestamp = int(datetime.now().timestamp() * 1000)
                
                while current_timestamp < end_timestamp:
                    try:
                        logger.debug(f"Fetching candles from {self.exchange.iso8601(current_timestamp)}")
                        # Most exchanges limit to 1000 candles per request
                        ohlcvs = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp, limit=1000)
                        
                        if not ohlcvs or len(ohlcvs) == 0:
                            logger.warning(f"No data returned at timestamp {current_timestamp}, stopping pagination")
                            break
                            
                        logger.debug(f"Fetched {len(ohlcvs)} candles")
                        
                        # Filter out any data beyond our end timestamp
                        if to_timestamp:
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
                        
                        # If we got 0 candles, we've likely reached the API limit or end of data
                        if len(ohlcvs) == 0:
                            logger.debug("Received zero candles, reached the end of available data")
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
            else:
                # Simple fetch with limit
                try:
                    logger.info(f"Fetching {limit} most recent candles")
                    ohlcvs = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    all_ohlcv = ohlcvs
                except Exception as e:
                    logger.error(f"Error fetching data: {e}")
                    logger.debug(traceback.format_exc())
            
            # Convert to DataFrame
            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Remove duplicates that might occur at pagination boundaries
                df = df[~df.index.duplicated(keep='first')]
                df = df.sort_index()
                
                logger.info(f"Successfully fetched {len(df)} records from {df.index[0]} to {df.index[-1]}")
                
                # Merge with cached data if we have any
                if market_data is not None:
                    logger.info("Merging new data with cached data")
                    combined_df = pd.concat([market_data, df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]  # Keep newer versions
                    combined_df = combined_df.sort_index()
                    
                    # Update the result
                    if limit and len(combined_df) > limit:
                        market_data = combined_df.iloc[-limit:]
                    else:
                        market_data = combined_df
                else:
                    # Use the fetched data directly
                    if limit and len(df) > limit:
                        market_data = df.iloc[-limit:]
                    else:
                        market_data = df
                
                # Update the cache
                if self.use_cache:
                    logger.debug(f"Updating OHLCV cache for {symbol}")
                    self.cache.update_ohlcv_cache(symbol, timeframe, df)
            else:
                # No data fetched, use whatever we had from cache
                logger.warning(f"No data fetched from exchange for {symbol}")
                if market_data is None:
                    logger.warning("No data available at all")
                    market_data = pd.DataFrame()
        
        # 4. If we have data but not indicators, and indicators were requested, calculate them
        if add_indicators and not market_data.empty:
            logger.info(f"Calculating technical indicators for {len(market_data)} candles")
            
            # Import here to avoid circular imports
            from quant_system.features.technical import TechnicalFeatures
            
            # Initialize with the same cache but force recalculation
            tech = TechnicalFeatures(cache=self.cache)
            
            # Force recalculation of indicators for the full dataset
            # First, calculate all indicators without going through cache
            df_indicators = tech._calculate_indicators(market_data)
            
            # Then update the cache with the new indicators
            if self.use_cache:
                logger.info(f"Updating indicators cache with {len(df_indicators)} records")
                self.cache.cache_indicators(symbol, timeframe, df_indicators)
            
            # Use the newly calculated indicators
            market_data = df_indicators
            
            logger.info(f"Added indicators to market data ({len(market_data.columns) - 5} indicators)")
        
        # 5. Export to CSV if requested
        if csv_output and not market_data.empty:
            try:
                logger.info(f"Saving market data to {csv_output}")
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(csv_output)), exist_ok=True)
                
                # Save to CSV
                market_data.to_csv(csv_output)
                logger.info(f"Successfully saved {len(market_data)} records to {csv_output}")
            except Exception as e:
                logger.error(f"Failed to save data to {csv_output}: {e}")
                logger.debug(traceback.format_exc())
        
        return market_data
        
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

    def fetch_multiple_symbols(self, symbols=['BTC/USD', 'ETH/USD'], timeframe='1d', limit=100, 
                            from_date=None, to_date=None, add_indicators=False):
        """Fetch market data for multiple symbols
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Candle timeframe
            limit: Optional number of candles to return (most recent)
            from_date: Optional start date
            to_date: Optional end date
            add_indicators: Whether to calculate and cache technical indicators
            
        Returns:
            Dictionary of symbol -> DataFrame with market data
        """
        logger.info(f"Fetching market data for {len(symbols)} symbols ({timeframe})")
        
        result = {}
        with ErrorHandler(context="fetching multiple symbols data") as handler:
            for symbol in symbols:
                logger.debug(f"Processing symbol: {symbol}")
                df = self.fetch_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    from_date=from_date,
                    to_date=to_date,
                    add_indicators=add_indicators
                )
                
                if not df.empty:
                    result[symbol] = df
                else:
                    logger.warning(f"No data available for {symbol}, skipping")
        
        logger.info(f"Successfully fetched data for {len(result)}/{len(symbols)} symbols")
        return result