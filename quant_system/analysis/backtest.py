import pandas as pd

class MarketBacktester:
    """Backtest performance following certain market conditions"""
    
    @staticmethod
    def find_similar_conditions(df, conditions, lookback_days=365):
        """Find historical periods with similar conditions"""
        similar_dates = []
        
        # Only use data within the lookback period
        lookback_df = df.iloc[-lookback_days:] if len(df) > lookback_days else df
        
        # Implement a simple pattern matching algorithm
        for i in range(len(lookback_df) - 1):
            date = lookback_df.index[i]
            current = lookback_df.iloc[i]
            prev = lookback_df.iloc[i-1] if i > 0 else None
            
            matched_conditions = []
            
            # Check for each condition
            if "Bullish trend" in conditions and current['close'] > current['sma_50'] and current['sma_50'] > current['sma_200']:
                matched_conditions.append("Bullish trend")
                
            if "Bearish trend" in conditions and current['close'] < current['sma_50'] and current['sma_50'] < current['sma_200']:
                matched_conditions.append("Bearish trend")
            
            # Check for other conditions similarly...
            if prev is not None:
                if "Golden cross" in conditions and prev['sma_50'] < prev['sma_200'] and current['sma_50'] > current['sma_200']:
                    matched_conditions.append("Golden cross")
                    
                if "Death cross" in conditions and prev['sma_50'] > prev['sma_200'] and current['sma_50'] < current['sma_200']:
                    matched_conditions.append("Death cross")
                
                if "MACD bullish crossover" in conditions and prev['macd'] < prev['macd_signal'] and current['macd'] > current['macd_signal']:
                    matched_conditions.append("MACD bullish crossover")
                    
                if "MACD bearish crossover" in conditions and prev['macd'] > prev['macd_signal'] and current['macd'] < current['macd_signal']:
                    matched_conditions.append("MACD bearish crossover")
            
            if "Oversold" in conditions and current['rsi_14'] < 30:
                matched_conditions.append("Oversold")
                
            if "Overbought" in conditions and current['rsi_14'] > 70:
                matched_conditions.append("Overbought")
            
            # If we've matched at least 2 conditions or a specific key condition
            if len(matched_conditions) >= 2 or any(cond in matched_conditions for cond in ["Golden cross", "Death cross"]):
                similar_dates.append((date, matched_conditions))
        
        return similar_dates
    
    @staticmethod
    def analyze_forward_returns(df, similar_dates, forward_days=[1, 5, 10, 20]):
        """Analyze returns following similar conditions"""
        results = []
        
        for date, conditions in similar_dates:
            date_loc = df.index.get_loc(date)
            
            # Skip if we don't have enough forward data
            if date_loc + max(forward_days) >= len(df):
                continue
            
            # Calculate forward returns
            base_price = df.iloc[date_loc]['close']
            returns = {}
            
            for days in forward_days:
                future_price = df.iloc[date_loc + days]['close']
                pct_return = (future_price - base_price) / base_price * 100
                returns[f"{days}d_return"] = pct_return
            
            results.append({
                'date': date,
                'conditions': conditions,
                **returns
            })
        
        # Convert to DataFrame for easy analysis
        results_df = pd.DataFrame(results)
        
        # Calculate stats
        stats = {}
        for days in forward_days:
            col = f"{days}d_return"
            if not results_df.empty and col in results_df.columns:
                stats[f"{days}d_mean"] = results_df[col].mean()
                stats[f"{days}d_median"] = results_df[col].median()
                stats[f"{days}d_positive_pct"] = (results_df[col] > 0).mean() * 100
        
        return results_df, stats
