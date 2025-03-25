import numpy as np
import pandas as pd
import ta

class TechnicalFeatures:
    """Generate technical indicators from price data"""
    
    @staticmethod
    def add_indicators(df):
        """Add common technical indicators to a dataframe"""
        if df.empty:
            return df
            
        # Copy the dataframe to avoid modifying the original
        df_indicators = df.copy()
        
        # Add moving averages
        df_indicators['sma_20'] = ta.trend.sma_indicator(df_indicators['close'], window=20)
        df_indicators['sma_50'] = ta.trend.sma_indicator(df_indicators['close'], window=50)
        df_indicators['sma_200'] = ta.trend.sma_indicator(df_indicators['close'], window=200)
        
        # Add RSI
        df_indicators['rsi_14'] = ta.momentum.rsi(df_indicators['close'], window=14)
        
        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df_indicators['close'], window=20, window_dev=2)
        df_indicators['bb_upper'] = bollinger.bollinger_hband()
        df_indicators['bb_lower'] = bollinger.bollinger_lband()
        
        # Add MACD
        macd = ta.trend.MACD(df_indicators['close'])
        df_indicators['macd'] = macd.macd()
        df_indicators['macd_signal'] = macd.macd_signal()
        
        # Distance from MA as Z-score
        df_indicators['z_score_ma_20'] = (df_indicators['close'] - df_indicators['sma_20']) / df_indicators['close'].rolling(20).std()
        
        # Volatility measures
        df_indicators['atr_14'] = ta.volatility.average_true_range(df_indicators['high'], df_indicators['low'], df_indicators['close'], window=14)
        df_indicators['volatility_20'] = df_indicators['close'].pct_change().rolling(20).std() * np.sqrt(20)
        
        return df_indicators
