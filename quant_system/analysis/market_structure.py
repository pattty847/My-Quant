# quant_system/analysis/market_structure.py
import numpy as np
import pandas as pd
from typing import List

from quant_system.utils import get_logger

# Initialize logger
logger = get_logger("analysis.market_structure")

class MarketStructureAnalyzer:
    """Analyze market structure and conditions"""
    
    @staticmethod
    def identify_conditions(df) -> List[str]:
        """Identify current market conditions based on indicators
        
        This method analyzes the technical indicators in the dataframe and
        returns a list of detected market conditions.
        
        Args:
            df: DataFrame with price data and technical indicators
            
        Returns:
            List of identified market conditions
        """
        if df.empty:
            logger.warning("Cannot identify conditions: Empty dataframe")
            return ["Error: Empty dataframe"]
            
        # Log the shape and available columns for debugging
        logger.debug(f"Identifying conditions from dataframe with {len(df)} rows")
        logger.debug(f"Available columns: {', '.join(df.columns)}")
        
        # Check we have enough data for analysis
        if len(df) < 20:
            logger.warning(f"Not enough data for analysis: {len(df)} rows")
            return [f"Insufficient data: Only {len(df)} candles (minimum 20 required)"]
            
        conditions = []
        
        # Get the most recent indicators
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        
        # Check for required columns and handle missing data
        required_columns = {
            'close': 'closing price',
            'high': 'high price',
            'low': 'low price',
            'volume': 'volume'
        }
        
        missing_columns = [f"{name} ({desc})" for name, desc in required_columns.items() 
                          if name not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required price columns: {', '.join(missing_columns)}")
            return [f"Missing essential data: {', '.join(missing_columns)}"]
        
        # Log latest price data
        logger.debug(f"Latest close: {latest['close']}, high: {latest['high']}, low: {latest['low']}")
        
        # ---------- TREND ANALYSIS ----------
        
        # Check for short-term moving averages (less dependent on long history)
        if 'sma_20' in df.columns and pd.notna(latest['sma_20']):
            if latest['close'] > latest['sma_20']:
                conditions.append("Price above 20-day MA (short-term bullish)")
            else:
                conditions.append("Price below 20-day MA (short-term bearish)")
        
        # Check for medium-term trend using 50-day MA if available
        if 'sma_50' in df.columns and pd.notna(latest['sma_50']):
            if latest['close'] > latest['sma_50']:
                conditions.append("Price above 50-day MA (medium-term bullish)")
            else:
                conditions.append("Price below 50-day MA (medium-term bearish)")
        
        # Use 200-day MA only if we have enough data
        if 'sma_200' in df.columns and pd.notna(latest['sma_200']):
            # Full trend analysis with 50 and 200 day MAs
            if 'sma_50' in df.columns and pd.notna(latest['sma_50']):
                if latest['close'] > latest['sma_50'] and latest['sma_50'] > latest['sma_200']:
                    conditions.append("Bullish trend (price above 50 & 200 MAs)")
                elif latest['close'] < latest['sma_50'] and latest['sma_50'] < latest['sma_200']:
                    conditions.append("Bearish trend (price below 50 & 200 MAs)")
            
            # Golden/Death cross detection (requires previous data point)
            if prev is not None and 'sma_50' in df.columns and 'sma_200' in df.columns:
                if pd.notna(prev['sma_50']) and pd.notna(prev['sma_200']):
                    if prev['sma_50'] < prev['sma_200'] and latest['sma_50'] > latest['sma_200']:
                        conditions.append("Golden cross (50 MA crossed above 200 MA)")
                    elif prev['sma_50'] > prev['sma_200'] and latest['sma_50'] < latest['sma_200']:
                        conditions.append("Death cross (50 MA crossed below 200 MA)")
        else:
            logger.debug("200-day MA not available or contains NaN values")
                    
        # ---------- MOMENTUM ANALYSIS ----------
        
        # RSI analysis
        if 'rsi_14' in df.columns and pd.notna(latest['rsi_14']):
            logger.debug(f"Latest RSI: {latest['rsi_14']}")
            
            if latest['rsi_14'] < 30:
                conditions.append("Oversold (RSI below 30)")
            elif latest['rsi_14'] > 70:
                conditions.append("Overbought (RSI above 70)")
            elif 30 <= latest['rsi_14'] < 40:
                conditions.append("Near oversold territory (RSI between 30-40)")
            elif 60 < latest['rsi_14'] <= 70:
                conditions.append("Near overbought territory (RSI between 60-70)")
        
        # MACD analysis
        if all(col in df.columns for col in ['macd', 'macd_signal']) and pd.notna(latest['macd']) and pd.notna(latest['macd_signal']):
            logger.debug(f"Latest MACD: {latest['macd']}, Signal: {latest['macd_signal']}")
            
            # Current MACD position
            if latest['macd'] > latest['macd_signal']:
                conditions.append("MACD above signal line (bullish)")
            else:
                conditions.append("MACD below signal line (bearish)")
                
            # MACD crossover (requires previous data point)
            if prev is not None and pd.notna(prev['macd']) and pd.notna(prev['macd_signal']):
                if prev['macd'] < prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
                    conditions.append("MACD bullish crossover")
                elif prev['macd'] > prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
                    conditions.append("MACD bearish crossover")
        
        # ---------- VOLATILITY ANALYSIS ----------
        
        # Bollinger Bands analysis
        bb_columns = ['bb_upper', 'bb_lower']
        if all(col in df.columns for col in bb_columns) and all(pd.notna(latest[col]) for col in bb_columns):
            logger.debug(f"Bollinger Bands - Upper: {latest['bb_upper']}, Lower: {latest['bb_lower']}")
            
            if latest['close'] < latest['bb_lower']:
                conditions.append("Price below lower Bollinger Band (potential reversal up)")
            elif latest['close'] > latest['bb_upper']:
                conditions.append("Price above upper Bollinger Band (potential reversal down)")
            
            # Calculate bandwidth if possible
            if 'sma_20' in df.columns and pd.notna(latest['sma_20']):
                bandwidth = (latest['bb_upper'] - latest['bb_lower']) / latest['sma_20']
                
                if bandwidth < 0.10:  # Threshold can be adjusted
                    conditions.append("Low volatility (narrow Bollinger Bands)")
                elif bandwidth > 0.40:  # Threshold can be adjusted
                    conditions.append("High volatility (wide Bollinger Bands)")
        
        # ATR for volatility regime
        if 'atr_14' in df.columns and pd.notna(latest['atr_14']):
            # Calculate ATR as percentage of price
            atr_pct = (latest['atr_14'] / latest['close']) * 100
            logger.debug(f"ATR: {latest['atr_14']}, ATR%: {atr_pct:.2f}%")
            
            if atr_pct < 1.5:
                conditions.append("Low volatility regime (ATR < 1.5% of price)")
            elif atr_pct > 5.0:
                conditions.append("High volatility regime (ATR > 5% of price)")
                
        # Explicit volatility calculation
        if 'volatility_20' in df.columns and pd.notna(latest['volatility_20']):
            logger.debug(f"20-day volatility: {latest['volatility_20']}")
            
            # Use percentiles of recent volatility to determine regime
            if len(df) >= 60:  # Need at least some history to calculate percentiles
                volatility_series = df['volatility_20'].dropna()
                percentiles = np.percentile(volatility_series, [25, 75])
                
                if latest['volatility_20'] < percentiles[0]:
                    conditions.append("Low volatility regime (bottom 25% of recent range)")
                elif latest['volatility_20'] > percentiles[1]:
                    conditions.append("High volatility regime (top 25% of recent range)")
            else:
                # Fallback with fixed thresholds if not enough history
                if latest['volatility_20'] < 0.015:  # 1.5% daily volatility
                    conditions.append("Low volatility")
                elif latest['volatility_20'] > 0.04:  # 4% daily volatility
                    conditions.append("High volatility")
        
        # ---------- PRICE ACTION ANALYSIS ----------
        
        # Recent price performance (5-day)
        if len(df) >= 5:
            five_day_change = ((latest['close'] / df.iloc[-5]['close']) - 1) * 100
            logger.debug(f"5-day price change: {five_day_change:.2f}%")
            
            if five_day_change > 10:
                conditions.append(f"Strong bullish move (+{five_day_change:.1f}% in 5 days)")
            elif five_day_change < -10:
                conditions.append(f"Strong bearish move ({five_day_change:.1f}% in 5 days)")
        
        # Volume analysis
        if 'volume' in df.columns:
            # Calculate average volume over last 20 days
            avg_volume = df['volume'].tail(20).mean()
            latest_volume = latest['volume']
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 0
            
            logger.debug(f"Latest volume: {latest_volume}, Avg volume: {avg_volume}, Ratio: {volume_ratio:.2f}")
            
            if volume_ratio > 2.0:
                conditions.append(f"Heavy volume ({volume_ratio:.1f}x average)")
            elif volume_ratio < 0.5:
                conditions.append(f"Light volume ({volume_ratio:.1f}x average)")
        
        # ---------- SUPPORT/RESISTANCE ANALYSIS ----------
        
        # Check for price near recent highs/lows
        if len(df) >= 20:
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            
            high_distance = (recent_high - latest['close']) / latest['close'] * 100
            low_distance = (latest['close'] - recent_low) / latest['close'] * 100
            
            logger.debug(f"Distance from 20-day high: {high_distance:.2f}%, from low: {low_distance:.2f}%")
            
            if high_distance < 1.0:
                conditions.append("Price near 20-day high (potential resistance)")
            elif low_distance < 1.0:
                conditions.append("Price near 20-day low (potential support)")
        
        # If we didn't identify any conditions (unusual but possible)
        if not conditions:
            logger.warning("No specific market conditions identified")
            conditions.append("Neutral market conditions")
        
        # Log all identified conditions
        logger.info(f"Identified {len(conditions)} market conditions: {', '.join(conditions)}")
        
        return conditions