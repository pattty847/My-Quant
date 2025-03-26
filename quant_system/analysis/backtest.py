# quant_system/analysis/backtest.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

from quant_system.utils import get_logger

# Initialize logger
logger = get_logger("analysis.backtest")

class MarketBacktester:
    """Backtest performance following certain market conditions"""
    
    @staticmethod
    def find_similar_conditions(df, conditions, lookback_days=None, min_match_count=1, exact_match=False):
        """Find historical periods with similar conditions
        
        Args:
            df: DataFrame with price data and indicators
            conditions: List of market conditions to search for
            lookback_days: Number of days to look back (None for all available data)
            min_match_count: Minimum number of matching conditions required
            exact_match: Whether all conditions must match exactly
            
        Returns:
            List of tuples with (date, matched_conditions)
        """
        if df.empty:
            logger.warning("Cannot find similar conditions: Empty dataframe")
            return []
        
        similar_dates = []
        condition_keywords = [cond.lower() for cond in conditions]
        
        # Determine lookback range
        if lookback_days is None:
            lookback_df = df.copy()
        else:
            # Don't look back more than the available data
            max_lookback = min(lookback_days, len(df) - 1)
            lookback_df = df.iloc[:-1].iloc[-max_lookback:]
        
        logger.info(f"Searching for similar conditions across {len(lookback_df)} historical data points")
        logger.debug(f"Target conditions: {', '.join(conditions)}")
        
        # Helper function to check if strings match based on keywords
        def conditions_match(target, reference):
            """Check if a condition matches another based on keywords"""
            target_lower = target.lower()
            reference_lower = reference.lower()
            
            # Check for exact matches first
            if target_lower == reference_lower:
                return True
                
            # Check for partial matches (one is substring of the other)
            if target_lower in reference_lower or reference_lower in target_lower:
                return True
                
            # Look for keyword matches
            target_words = set(target_lower.split())
            reference_words = set(reference_lower.split())
            
            # If they share at least 3 significant words, consider it a match
            shared_words = target_words.intersection(reference_words)
            significant_words = [w for w in shared_words if len(w) > 3 and w not in {'with', 'from', 'that', 'this', 'above', 'below', 'near'}]
            
            return len(significant_words) >= 2
        
        # Process each historical data point
        for i in range(len(lookback_df)):
            date = lookback_df.index[i]
            current = lookback_df.iloc[i]
            prev = lookback_df.iloc[i-1] if i > 0 else None
            
            # Skip rows with NaN values in key columns
            if pd.isna(current['close']):
                continue
                
            # Identify conditions at this historical point
            historical_conditions = MarketBacktester._identify_historical_conditions(lookback_df, i, prev)
            
            # Match conditions between current target and historical point
            matched_conditions = []
            
            for hist_cond in historical_conditions:
                # For each current condition we're looking for
                for target_cond in conditions:
                    if conditions_match(target_cond, hist_cond):
                        matched_conditions.append(hist_cond)
                        break
            
            # Check if we have enough matches
            if len(matched_conditions) >= min_match_count:
                if exact_match and len(matched_conditions) < len(conditions):
                    # Skip if exact match required but not all conditions matched
                    continue
                    
                similar_dates.append((date, matched_conditions))
                logger.debug(f"Found match at {date.date()}: {', '.join(matched_conditions)}")
        
        logger.info(f"Found {len(similar_dates)} similar historical instances")
        return similar_dates
    
    @staticmethod
    def _identify_historical_conditions(df, idx, prev):
        """Identify market conditions at a historical data point
        
        This is a simplified version of MarketStructureAnalyzer.identify_conditions
        that works on a specific row in a dataframe.
        
        Args:
            df: DataFrame with price data and indicators
            idx: Index of the row to analyze
            prev: Previous row for calculations requiring it
            
        Returns:
            List of identified conditions
        """
        conditions = []
        current = df.iloc[idx]
        
        # ---------- TREND CONDITIONS ----------
        
        # Moving Average relationships
        if 'sma_20' in df.columns and pd.notna(current['sma_20']):
            if current['close'] > current['sma_20']:
                conditions.append("Price above 20-day MA")
            else:
                conditions.append("Price below 20-day MA")
        
        if 'sma_50' in df.columns and pd.notna(current['sma_50']):
            if current['close'] > current['sma_50']:
                conditions.append("Price above 50-day MA")
            else:
                conditions.append("Price below 50-day MA")
        
        if 'sma_200' in df.columns and pd.notna(current['sma_200']):
            if current['close'] > current['sma_200']:
                conditions.append("Price above 200-day MA")
            else:
                conditions.append("Price below 200-day MA")
            
            # Full trend with 50-day MA
            if 'sma_50' in df.columns and pd.notna(current['sma_50']):
                if current['close'] > current['sma_50'] and current['sma_50'] > current['sma_200']:
                    conditions.append("Bullish trend")
                elif current['close'] < current['sma_50'] and current['sma_50'] < current['sma_200']:
                    conditions.append("Bearish trend")
            
            # Golden/Death cross
            if prev is not None and 'sma_50' in df.columns and 'sma_200' in df.columns:
                if pd.notna(prev['sma_50']) and pd.notna(prev['sma_200']):
                    if prev['sma_50'] < prev['sma_200'] and current['sma_50'] > current['sma_200']:
                        conditions.append("Golden cross")
                    elif prev['sma_50'] > prev['sma_200'] and current['sma_50'] < current['sma_200']:
                        conditions.append("Death cross")
        
        # ---------- MOMENTUM CONDITIONS ----------
        
        # RSI
        if 'rsi_14' in df.columns and pd.notna(current['rsi_14']):
            if current['rsi_14'] < 30:
                conditions.append("Oversold")
            elif current['rsi_14'] > 70:
                conditions.append("Overbought")
        
        # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']) and pd.notna(current['macd']) and pd.notna(current['macd_signal']):
            if current['macd'] > current['macd_signal']:
                conditions.append("MACD above signal")
            else:
                conditions.append("MACD below signal")
                
            if prev is not None and pd.notna(prev['macd']) and pd.notna(prev['macd_signal']):
                if prev['macd'] < prev['macd_signal'] and current['macd'] > current['macd_signal']:
                    conditions.append("MACD bullish crossover")
                elif prev['macd'] > prev['macd_signal'] and current['macd'] < current['macd_signal']:
                    conditions.append("MACD bearish crossover")
        
        # ---------- VOLATILITY CONDITIONS ----------
        
        # Bollinger Bands
        bb_columns = ['bb_upper', 'bb_lower']
        if all(col in df.columns for col in bb_columns) and all(pd.notna(current[col]) for col in bb_columns):
            if current['close'] < current['bb_lower']:
                conditions.append("Price below lower Bollinger Band")
            elif current['close'] > current['bb_upper']:
                conditions.append("Price above upper Bollinger Band")
        
        # ATR
        if 'atr_14' in df.columns and pd.notna(current['atr_14']) and current['close'] > 0:
            atr_pct = (current['atr_14'] / current['close']) * 100
            
            if atr_pct < 1.5:
                conditions.append("Low volatility")
            elif atr_pct > 5.0:
                conditions.append("High volatility")
        
        # ---------- PRICE ACTION ----------
        
        # Recent price changes can be calculated even with minimal data
        lookback = min(5, idx)
        if lookback > 0 and pd.notna(df.iloc[idx-lookback]['close']):
            price_change = ((current['close'] / df.iloc[idx-lookback]['close']) - 1) * 100
            
            if price_change > 10:
                conditions.append("Strong bullish move")
            elif price_change < -10:
                conditions.append("Strong bearish move")
        
        return conditions
    
    @staticmethod
    def analyze_forward_returns(df, similar_dates, forward_days=[1, 5, 10, 20]):
        """Analyze returns following similar conditions
        
        Args:
            df: DataFrame with price data
            similar_dates: List of tuples with (date, matched_conditions)
            forward_days: List of forward periods to analyze
            
        Returns:
            (results_df, stats) - DataFrame with detailed results and dictionary with stats
        """
        if not similar_dates:
            logger.warning("No similar dates provided for forward returns analysis")
            return pd.DataFrame(), {}
        
        results = []
        
        logger.info(f"Analyzing forward returns for {len(similar_dates)} instances")
        logger.debug(f"Forward periods: {forward_days} days")
        
        for date, conditions in similar_dates:
            try:
                date_loc = df.index.get_loc(date)
                
                # Skip if we don't have enough forward data
                max_forward = max(forward_days)
                if date_loc + max_forward >= len(df):
                    logger.debug(f"Skipping {date.date()}: insufficient forward data")
                    continue
                
                # Calculate forward returns
                base_price = df.iloc[date_loc]['close']
                returns = {}
                
                for days in forward_days:
                    # Ensure we have data at the forward date
                    if date_loc + days < len(df) and pd.notna(df.iloc[date_loc + days]['close']):
                        future_price = df.iloc[date_loc + days]['close']
                        pct_return = (future_price - base_price) / base_price * 100
                        returns[f"{days}d_return"] = pct_return
                    else:
                        logger.debug(f"Missing {days}-day forward data for {date.date()}")
                
                # Add record if we have at least one valid return
                if returns:
                    results.append({
                        'date': date,
                        'conditions': conditions,
                        **returns
                    })
            except Exception as e:
                logger.error(f"Error calculating forward returns for {date}: {e}")
        
        # Convert to DataFrame for easy analysis
        results_df = pd.DataFrame(results) if results else pd.DataFrame()
        
        # Calculate stats
        stats = {}
        if not results_df.empty:
            for days in forward_days:
                col = f"{days}d_return"
                if col in results_df.columns:
                    valid_returns = results_df[col].dropna()
                    
                    if not valid_returns.empty:
                        stats[f"{days}d_mean"] = valid_returns.mean()
                        stats[f"{days}d_median"] = valid_returns.median()
                        stats[f"{days}d_std"] = valid_returns.std()
                        stats[f"{days}d_min"] = valid_returns.min()
                        stats[f"{days}d_max"] = valid_returns.max()
                        stats[f"{days}d_positive_pct"] = (valid_returns > 0).mean() * 100
                        stats[f"{days}d_count"] = len(valid_returns)
                        
                        logger.info(f"{days}-day forward returns: mean={stats[f'{days}d_mean']:.2f}%, "
                                  f"win rate={stats[f'{days}d_positive_pct']:.2f}%")
        else:
            logger.warning("No valid forward returns calculated")
        
        return results_df, stats