# quant_system/features/technical.py
import numpy as np
import pandas as pd
import ta
import math
from datetime import datetime, timedelta

from quant_system.utils import get_logger

# Initialize logger
logger = get_logger("features.technical")

class TechnicalFeatures:
    """Generate technical indicators from price data"""
    
    @staticmethod
    def add_indicators(df, required_periods=None):
        """Add common technical indicators to a dataframe
        
        Adapts the indicators based on available data length.
        
        Args:
            df: DataFrame with OHLCV data
            required_periods: Optional list of periods to ensure are calculated
            
        Returns:
            DataFrame with technical indicators added
        """
        if df.empty:
            logger.warning("Cannot add indicators: Empty dataframe")
            return df
        
        data_length = len(df)
        logger.info(f"Adding technical indicators to dataframe with {data_length} rows")
        
        # Copy the dataframe to avoid modifying the original
        df_indicators = df.copy()
        
        # Log the date range
        start_date = df_indicators.index[0]
        end_date = df_indicators.index[-1]
        logger.debug(f"Data range: {start_date.date()} to {end_date.date()}")
        
        # Determine maximum reasonable periods based on data length
        # Rule of thumb: Don't use MA periods > data_length/2
        max_ma_period = min(200, math.floor(data_length/2))
        
        # If we have very limited data, adjust all periods proportionally
        if data_length < 50:
            logger.warning(f"Limited data available ({data_length} rows), scaling indicator periods")
            scale_factor = data_length / 100  # Scale based on ideal 100+ data points
        else:
            scale_factor = 1.0
            
        logger.debug(f"Using max MA period of {max_ma_period}, scale factor {scale_factor:.2f}")
        
        # ---------- MOVING AVERAGES ----------
        
        # Add moving averages based on available data
        ma_periods = [5, 10, 20, 50, 200]
        ma_periods = [p for p in ma_periods if p <= max_ma_period]
        
        for period in ma_periods:
            adjusted_period = max(2, int(period * scale_factor))
            logger.debug(f"Calculating SMA with period {adjusted_period} (original: {period})")
            
            df_indicators[f'sma_{period}'] = ta.trend.sma_indicator(
                df_indicators['close'], 
                window=adjusted_period
            )
        
        # ---------- MOMENTUM INDICATORS ----------
        
        # RSI (default 14, min 2)
        rsi_period = max(2, int(14 * scale_factor))
        logger.debug(f"Calculating RSI with period {rsi_period}")
        df_indicators['rsi_14'] = ta.momentum.rsi(df_indicators['close'], window=rsi_period)
        
        # MACD (default 12/26/9)
        # Adjust if we have limited data
        if data_length >= 30:
            fast_period = int(12 * scale_factor)
            slow_period = int(26 * scale_factor)
            signal_period = int(9 * scale_factor)
            
            # Ensure minimum values
            fast_period = max(2, fast_period)
            slow_period = max(fast_period + 1, slow_period)
            signal_period = max(2, signal_period)
            
            logger.debug(f"Calculating MACD with periods {fast_period}/{slow_period}/{signal_period}")
            
            macd = ta.trend.MACD(
                df_indicators['close'], 
                window_fast=fast_period, 
                window_slow=slow_period, 
                window_sign=signal_period
            )
            df_indicators['macd'] = macd.macd()
            df_indicators['macd_signal'] = macd.macd_signal()
            df_indicators['macd_histogram'] = macd.macd_diff()
        else:
            logger.warning(f"Insufficient data for MACD calculation (need 30+, have {data_length})")
        
        # ---------- VOLATILITY INDICATORS ----------
        
        # Bollinger Bands (default 20, min 2)
        bb_period = max(2, int(20 * scale_factor))
        logger.debug(f"Calculating Bollinger Bands with period {bb_period}")
        
        bollinger = ta.volatility.BollingerBands(
            df_indicators['close'], 
            window=bb_period, 
            window_dev=2
        )
        df_indicators['bb_upper'] = bollinger.bollinger_hband()
        df_indicators['bb_lower'] = bollinger.bollinger_lband()
        df_indicators['bb_middle'] = bollinger.bollinger_mavg()
        df_indicators['bb_width'] = (df_indicators['bb_upper'] - df_indicators['bb_lower']) / df_indicators['bb_middle']
        
        # ATR (default 14, min 2)
        atr_period = max(2, int(14 * scale_factor))
        logger.debug(f"Calculating ATR with period {atr_period}")
        
        df_indicators['atr_14'] = ta.volatility.average_true_range(
            df_indicators['high'], 
            df_indicators['low'], 
            df_indicators['close'], 
            window=atr_period
        )
        
        # Volatility (std dev of returns)
        vol_period = max(2, int(20 * scale_factor))
        logger.debug(f"Calculating volatility with period {vol_period}")
        
        df_indicators['volatility_20'] = df_indicators['close'].pct_change().rolling(vol_period).std() * np.sqrt(252 / periodicity_factor(df))
        
        # ---------- Z-SCORES & OSCILLATORS ----------
        
        # Z-score of price distance from MA (for short-term MAs only)
        for period in [20, 50]:
            if f'sma_{period}' in df_indicators.columns:
                ma_col = f'sma_{period}'
                z_col = f'z_score_ma_{period}'
                
                # Calculate z-score if we have enough data
                if data_length >= period * 1.5:
                    logger.debug(f"Calculating Z-score for {ma_col}")
                    std_period = max(5, int(period / 2))
                    df_indicators[z_col] = (df_indicators['close'] - df_indicators[ma_col]) / df_indicators['close'].rolling(std_period).std()
        
        # Stochastic Oscillator if we have enough data
        if data_length >= 14:
            stoch_period = max(5, int(14 * scale_factor))
            stoch_smooth = max(3, int(3 * scale_factor))
            
            logger.debug(f"Calculating Stochastic Oscillator with periods {stoch_period}/{stoch_smooth}")
            
            stoch = ta.momentum.StochasticOscillator(
                df_indicators['high'], 
                df_indicators['low'], 
                df_indicators['close'], 
                window=stoch_period, 
                smooth_window=stoch_smooth
            )
            df_indicators['stoch_k'] = stoch.stoch()
            df_indicators['stoch_d'] = stoch.stoch_signal()
        
        # ---------- TREND INDICATORS ----------
        
        # ADX if we have enough data
        if data_length >= 28:
            adx_period = max(14, int(14 * scale_factor))
            logger.debug(f"Calculating ADX with period {adx_period}")
            
            adx = ta.trend.ADXIndicator(
                df_indicators['high'], 
                df_indicators['low'], 
                df_indicators['close'], 
                window=adx_period
            )
            df_indicators['adx'] = adx.adx()
            df_indicators['di_plus'] = adx.adx_pos()
            df_indicators['di_minus'] = adx.adx_neg()
        
        # ---------- CUSTOM INDICATORS ----------
        
        # Calculate some basic stats about the data
        df_indicators['returns_1d'] = df_indicators['close'].pct_change()
        
        # Calculate distance from recent highs/lows if we have enough data
        if data_length >= 20:
            logger.debug("Calculating high/low metrics")
            df_indicators['dist_from_20d_high'] = df_indicators['close'] / df_indicators['high'].rolling(20).max() - 1
            df_indicators['dist_from_20d_low'] = df_indicators['close'] / df_indicators['low'].rolling(20).min() - 1
            
            # Calculate days since last N-day high/low
            for period in [20, 50]:
                if data_length >= period:
                    # Rolling max/min with expanding window method
                    rolling_max = df_indicators['high'].rolling(period).max()
                    rolling_min = df_indicators['low'].rolling(period).min()
                    
                    # Initialize counters
                    days_since_high = np.zeros(len(df_indicators))
                    days_since_low = np.zeros(len(df_indicators))
                    
                    # Calculate days since high/low
                    for i in range(period, len(df_indicators)):
                        if df_indicators['high'].iloc[i] >= rolling_max.iloc[i-1]:
                            # New high
                            days_since_high[i] = 0
                        else:
                            # Increment counter
                            days_since_high[i] = days_since_high[i-1] + 1
                            
                        if df_indicators['low'].iloc[i] <= rolling_min.iloc[i-1]:
                            # New low
                            days_since_low[i] = 0
                        else:
                            # Increment counter
                            days_since_low[i] = days_since_low[i-1] + 1
                    
                    df_indicators[f'days_since_{period}d_high'] = days_since_high
                    df_indicators[f'days_since_{period}d_low'] = days_since_low
        
        # Log count of generated indicators
        num_indicators = len(df_indicators.columns) - len(df.columns)
        logger.info(f"Added {num_indicators} technical indicators")
        
        return df_indicators

def periodicity_factor(df):
    """Determine the periodicity factor for annualizing volatility
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        Factor to use for annualizing (252 for daily, 52 for weekly, etc.)
    """
    if len(df) < 2:
        return 252  # Default to daily
        
    # Calculate average time delta between entries
    deltas = []
    for i in range(1, min(len(df), 10)):
        delta = (df.index[i] - df.index[i-1]).total_seconds()
        deltas.append(delta)
    
    avg_seconds = sum(deltas) / len(deltas)
    
    # Determine periodicity
    if avg_seconds < 3600:  # Less than 1 hour
        minutes = avg_seconds / 60
        logger.debug(f"Detected {minutes:.1f} minute data")
        return 252 * 6.5 * (60 / minutes)  # Assuming 6.5 trading hours
    elif avg_seconds < 3600 * 24:  # Less than 1 day
        hours = avg_seconds / 3600
        logger.debug(f"Detected {hours:.1f} hour data")
        return 252 * (24 / hours)
    elif avg_seconds < 3600 * 24 * 7:  # Less than 1 week
        days = avg_seconds / (3600 * 24)
        logger.debug(f"Detected {days:.1f} day data")
        return 252 / days
    elif avg_seconds < 3600 * 24 * 30:  # Less than 1 month
        weeks = avg_seconds / (3600 * 24 * 7)
        logger.debug(f"Detected {weeks:.1f} week data")
        return 52 / weeks
    else:  # Monthly or longer
        months = avg_seconds / (3600 * 24 * 30)
        logger.debug(f"Detected {months:.1f} month data")
        return 12 / months