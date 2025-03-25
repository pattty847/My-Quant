# interface/cli.py
import argparse
import json
import os
import sys
from typing import List, Dict, Any
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our system components
from quant_system.data.connectors import CryptoDataConnector
from quant_system.features.technical import TechnicalFeatures
from quant_system.analysis.market_structure import MarketStructureAnalyzer
from quant_system.analysis.backtest import MarketBacktester
from quant_system.analysis.llm_interface import LLMAnalyzer

def pretty_print_json(data):
    """Print JSON data in a human-readable format"""
    print(json.dumps(data, indent=2))

class QuantSystemCLI:
    """Command-line interface for the Quant System"""
    
    def __init__(self):
        self.data_connector = CryptoDataConnector()
        self.technical_features = TechnicalFeatures()
        self.market_analyzer = MarketStructureAnalyzer()
        self.backtester = MarketBacktester()
        self.llm = LLMAnalyzer(api_key=os.environ.get("LLM_API_KEY"))
    
    def analyze(self, symbol: str, timeframe: str, days: int):
        """Run a complete market analysis"""
        print(f"Analyzing {symbol} on {timeframe} timeframe using {days} days of data...\n")
        
        # 1. Fetch data
        market_data = self.data_connector.fetch_ohlcv(symbol, timeframe, limit=days)
        
        if market_data.empty:
            print("Error: Could not fetch market data")
            return 1
        
        print(f"Data retrieved: {len(market_data)} candles from {market_data.index[0]} to {market_data.index[-1]}\n")
        
        # 2. Generate technical features
        df_indicators = self.technical_features.add_indicators(market_data)
        
        # 3. Identify current market conditions
        conditions = self.market_analyzer.identify_conditions(df_indicators)
        
        print("Current Market Conditions:")
        for i, condition in enumerate(conditions, 1):
            print(f"  {i}. {condition}")
        print()
        
        # 4. Find similar historical conditions and analyze performance
        similar_dates = self.backtester.find_similar_conditions(df_indicators, conditions)
        
        if not similar_dates:
            print("No similar historical conditions found in the specified period.")
            return 0
        
        print(f"Found {len(similar_dates)} similar historical instances")
        print("Recent examples:")
        for date, matched_conditions in similar_dates[-3:]:
            print(f"  {date.date()}: {', '.join(matched_conditions)}")
        print()
        
        # 5. Calculate forward returns from similar conditions
        results_df, stats = self.backtester.analyze_forward_returns(df_indicators, similar_dates)
        
        print("Historical Forward Returns (after similar conditions):")
        for period, value in stats.items():
            if 'mean' in period:
                days = period.split('d_')[0]
                print(f"  {days}-day forward return (avg): {value:.2f}%")
            if 'positive_pct' in period:
                days = period.split('d_')[0]
                print(f"  {days}-day win rate: {value:.2f}%")
        print()
        
        # 6. Generate market summary with LLM
        if os.environ.get("LLM_API_KEY"):
            print("Generating market analysis with LLM...")
            summary = self.llm.generate_market_summary(df_indicators, conditions, stats)
            print("\nLLM Analysis:")
            print(summary)
        else:
            print("\nLLM Analysis skipped (no API key provided)")
        
        return 0
    
    def list_symbols(self):
        """List available trading pairs"""
        try:
            exchange = self.data_connector.exchange
            markets = exchange.load_markets()
            symbols = list(markets.keys())
            
            print(f"Available symbols on {exchange.name}:")
            
            # Group by base currency
            by_base = {}
            for symbol in symbols:
                parts = symbol.split('/')
                if len(parts) == 2:
                    base = parts[0]
                    if base not in by_base:
                        by_base[base] = []
                    by_base[base].append(symbol)
            
            # Print major currencies first
            major_currencies = ['BTC', 'ETH', 'USDT', 'USDC', 'BNB', 'SOL']
            for currency in major_currencies:
                if currency in by_base:
                    pairs = by_base[currency]
                    print(f"  {currency}: {', '.join(pairs[:5])}{'...' if len(pairs) > 5 else ''}")
                    del by_base[currency]
            
            # Print remaining currencies
            for currency, pairs in list(by_base.items())[:10]:  # Limit to avoid too much output
                print(f"  {currency}: {', '.join(pairs[:5])}{'...' if len(pairs) > 5 else ''}")
            
            if len(by_base) > 10:
                print(f"  ... and {len(by_base) - 10} more base currencies")
                
            return 0
        except Exception as e:
            print(f"Error listing symbols: {str(e)}")
            return 1
    
    def list_timeframes(self):
        """List available timeframes"""
        try:
            exchange = self.data_connector.exchange
            timeframes = exchange.timeframes
            
            print(f"Available timeframes on {exchange.name}:")
            for tf, description in timeframes.items():
                print(f"  {tf}: {description}")
            
            return 0
        except Exception as e:
            print(f"Error listing timeframes: {str(e)}")
            return 1
    
    def backtest(self, conditions: List[str], symbol: str, timeframe: str, days: int):
        """Backtest specific market conditions"""
        print(f"Backtesting conditions: {', '.join(conditions)}")
        print(f"Symbol: {symbol}, Timeframe: {timeframe}, Data period: {days} days\n")
        
        # 1. Fetch data
        market_data = self.data_connector.fetch_ohlcv(symbol, timeframe, limit=days)
        
        if market_data.empty:
            print("Error: Could not fetch market data")
            return 1
        
        # 2. Generate technical features
        df_indicators = self.technical_features.add_indicators(market_data)
        
        # 3. Find similar dates
        similar_dates = self.backtester.find_similar_conditions(df_indicators, conditions)
        
        if not similar_dates:
            print("No instances of these conditions found in the historical data.")
            return 0
        
        print(f"Found {len(similar_dates)} instances of specified conditions")
        
        # 4. Calculate forward returns
        results_df, stats = self.backtester.analyze_forward_returns(df_indicators, similar_dates)
        
        print("\nHistorical Forward Returns:")
        for period, value in stats.items():
            if 'mean' in period:
                days = period.split('d_')[0]
                print(f"  {days}-day forward return (avg): {value:.2f}%")
            if 'positive_pct' in period:
                days = period.split('d_')[0]
                print(f"  {days}-day win rate: {value:.2f}%")
        
        if not results_df.empty:
            print("\nMost Recent Occurrences:")
            recent = results_df.sort_values('date', ascending=False).head(5)
            for _, row in recent.iterrows():
                date = row['date'].date()
                conditions_str = ', '.join(row['conditions'])
                returns = [f"{days}d: {row[f'{days}d_return']:.2f}%" 
                           for days in [1, 5, 10, 20] 
                           if f"{days}d_return" in row and not pd.isna(row[f"{days}d_return"])]
                
                print(f"  {date} - {conditions_str}")
                print(f"    Returns: {', '.join(returns)}")
        
        return 0

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="Quant System Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run complete market analysis")
    analyze_parser.add_argument("--symbol", "-s", default="BTC/USDT", help="Trading pair symbol")
    analyze_parser.add_argument("--timeframe", "-t", default="1d", help="Timeframe (e.g., 1m, 1h, 1d)")
    analyze_parser.add_argument("--days", "-d", type=int, default=365, help="Days of historical data")
    
    # List symbols command
    subparsers.add_parser("list-symbols", help="List available trading pairs")
    
    # List timeframes command
    subparsers.add_parser("list-timeframes", help="List available timeframes")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Backtest specific market conditions")
    backtest_parser.add_argument("conditions", nargs="+", help="Market conditions to backtest")
    backtest_parser.add_argument("--symbol", "-s", default="BTC/USDT", help="Trading pair symbol")
    backtest_parser.add_argument("--timeframe", "-t", default="1d", help="Timeframe (e.g., 1m, 1h, 1d)")
    backtest_parser.add_argument("--days", "-d", type=int, default=365, help="Days of historical data")
    
    args = parser.parse_args()
    cli = QuantSystemCLI()
    
    if args.command == "analyze":
        return cli.analyze(args.symbol, args.timeframe, args.days)
    elif args.command == "list-symbols":
        return cli.list_symbols()
    elif args.command == "list-timeframes":
        return cli.list_timeframes()
    elif args.command == "backtest":
        return cli.backtest(args.conditions, args.symbol, args.timeframe, args.days)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())