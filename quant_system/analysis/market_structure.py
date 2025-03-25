import numpy as np
import pandas as pd

class MarketStructureAnalyzer:
    """Analyze market structure and conditions"""
    
    @staticmethod
    def identify_conditions(df):
        """Identify current market conditions based on indicators"""
        if df.empty or len(df) < 50:
            return "Insufficient data for analysis"
            
        conditions = []
        
        # Get the most recent indicators
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Trend analysis
        if latest['close'] > latest['sma_50'] and latest['sma_50'] > latest['sma_200']:
            conditions.append("Bullish trend (price above 50 & 200 SMA)")
        elif latest['close'] < latest['sma_50'] and latest['sma_50'] < latest['sma_200']:
            conditions.append("Bearish trend (price below 50 & 200 SMA)")
        
        # MA crossovers
        if prev['sma_50'] < prev['sma_200'] and latest['sma_50'] > latest['sma_200']:
            conditions.append("Golden cross (50 SMA crossed above 200 SMA)")
        elif prev['sma_50'] > prev['sma_200'] and latest['sma_50'] < latest['sma_200']:
            conditions.append("Death cross (50 SMA crossed below 200 SMA)")
        
        # Oversold/Overbought
        if latest['rsi_14'] < 30:
            conditions.append("Oversold (RSI below 30)")
        elif latest['rsi_14'] > 70:
            conditions.append("Overbought (RSI above 70)")
        
        # Volatility state
        volatility_percentile = np.percentile(df['volatility_20'].dropna(), [25, 50, 75])
        if latest['volatility_20'] < volatility_percentile[0]:
            conditions.append("Low volatility regime")
        elif latest['volatility_20'] > volatility_percentile[2]:
            conditions.append("High volatility regime")
        
        # Bollinger Band signals
        if latest['close'] < latest['bb_lower']:
            conditions.append("Price below lower Bollinger Band")
        elif latest['close'] > latest['bb_upper']:
            conditions.append("Price above upper Bollinger Band")
        
        # MACD signals
        if prev['macd'] < prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
            conditions.append("MACD bullish crossover")
        elif prev['macd'] > prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
            conditions.append("MACD bearish crossover")
        
        # Return all identified conditions
        return conditions
