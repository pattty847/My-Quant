# interface/cli.py
import argparse
import json
import os
import sys
import traceback
from typing import List, Dict, Any
import pandas as pd
import time
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our system components
from quant_system.data.connectors import CryptoDataConnector
from quant_system.features.technical import TechnicalFeatures
from quant_system.analysis.market_structure import MarketStructureAnalyzer
from quant_system.analysis.backtest import MarketBacktester
from quant_system.analysis.llm_interface import LLMAnalyzer
from quant_system.analysis.advanced_models import run_models, DimensionalityReducer, RegimeDetector
from quant_system.utils import get_logger, ErrorHandler, set_log_level, DEBUG, INFO

# Initialize logger for this module
logger = get_logger("interface.cli")

def pretty_print_json(data):
    """Print JSON data in a human-readable format"""
    print(json.dumps(data, indent=2))

class QuantSystemCLI:
    """Command-line interface for the Quant System"""
    
    def __init__(self):
        logger.debug("Initializing CLI interface")
        try:
            self.data_connector = CryptoDataConnector()
            self.technical_features = TechnicalFeatures(cache=self.data_connector.cache)
            self.market_analyzer = MarketStructureAnalyzer()
            self.backtester = MarketBacktester()
            self.llm = LLMAnalyzer(api_key=os.environ.get("LLM_API_KEY"))
            logger.debug("CLI system components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CLI system components: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def analyze(self, symbol: str, timeframe: str, days: int):
        """Run a complete market analysis"""
        print(f"Analyzing {symbol} on {timeframe} timeframe using {days} days of data...\n")
        logger.info(f"Starting analysis of {symbol} on {timeframe} timeframe ({days} days)")
        
        with ErrorHandler(context=f"market analysis for {symbol}") as handler:
            # 1. Fetch data
            logger.debug(f"Fetching market data for {symbol}")
            market_data = self.data_connector.fetch_market_data(
                symbol=symbol, 
                timeframe=timeframe, 
                limit=days
            )
            
            if market_data.empty:
                error_msg = f"Could not fetch market data for {symbol}"
                logger.error(error_msg)
                print(f"Error: {error_msg}")
                return 1
            
            logger.info(f"Retrieved {len(market_data)} candles from {market_data.index[0]} to {market_data.index[-1]}")
            
            # 2. Generate technical features
            logger.debug(f"Generating technical indicators for {symbol}")
            df_indicators = self.technical_features.add_indicators(market_data, symbol, timeframe)
            
            # 3. Identify current market conditions
            logger.debug(f"Identifying market conditions for {symbol}")
            conditions = self.market_analyzer.identify_conditions(df_indicators)
            logger.info(f"Identified conditions for {symbol}: {', '.join(conditions)}")
            
            print("Current Market Conditions:")
            for i, condition in enumerate(conditions, 1):
                print(f"  {i}. {condition}")
            print()
            
            # 4. Find similar historical conditions and analyze performance
            logger.debug(f"Finding similar historical conditions for {symbol}")
            similar_dates = self.backtester.find_similar_conditions(df_indicators, conditions)
            
            if not similar_dates:
                logger.info(f"No similar historical conditions found for {symbol}")
                print("No similar historical conditions found in the specified period.")
                return 0
            
            logger.info(f"Found {len(similar_dates)} similar historical instances for {symbol}")
            print(f"Found {len(similar_dates)} similar historical instances")
            print("Recent examples:")
            for date, matched_conditions in similar_dates[-3:]:
                print(f"  {date.date()}: {', '.join(matched_conditions)}")
            print()
            
            # 5. Calculate forward returns from similar conditions
            logger.debug(f"Calculating forward returns for {symbol}")
            results_df, stats = self.backtester.analyze_forward_returns(df_indicators, similar_dates)
            
            print("Historical Forward Returns (after similar conditions):")
            for period, value in stats.items():
                if 'mean' in period:
                    days = period.split('d_')[0]
                    logger.info(f"{days}-day forward return (avg): {value:.2f}%")
                    print(f"  {days}-day forward return (avg): {value:.2f}%")
                if 'positive_pct' in period:
                    days = period.split('d_')[0]
                    logger.info(f"{days}-day win rate: {value:.2f}%")
                    print(f"  {days}-day win rate: {value:.2f}%")
            print()
            
            # 6. Generate market summary with LLM
            if os.environ.get("LLM_API_KEY"):
                logger.debug(f"Generating LLM market analysis for {symbol}")
                print("Generating market analysis with LLM...")
                summary = self.llm.generate_market_summary(df_indicators, conditions, stats)
                logger.info(f"LLM analysis generated for {symbol}")
                print("\nLLM Analysis:")
                print(summary)
            else:
                logger.warning("LLM analysis skipped (no API key provided)")
                print("\nLLM Analysis skipped (no API key provided)")
        
        logger.info(f"Analysis of {symbol} completed successfully")
        return 0
    
    def analyze_advanced(self, symbol: str, timeframe: str, days: int, output_dir: str = None, 
                         n_components: int = 2, n_regimes: int = 3, force_refresh: bool = False):
        """Run advanced statistical analysis (PCA & HMM)"""
        print(f"Running advanced analysis for {symbol} on {timeframe} timeframe using {days} days of data...")
        print(f"PCA components: {n_components}, Regime clusters: {n_regimes}\n")
        
        logger.info(f"Starting advanced analysis of {symbol} on {timeframe} timeframe ({days} days)")
        
        # Create output directory based on symbol and date if not provided
        if not output_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            symbol_formatted = symbol.replace('/', '_')
            output_dir = f"./analysis_results/{symbol_formatted}_{timeframe}_{timestamp}"
            
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        
        with ErrorHandler(context=f"advanced analysis for {symbol}") as handler:
            # 1. Fetch data - use from_date instead of days to get full historical data
            logger.debug(f"Fetching market data for {symbol}")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            logger.info(f"Fetching data from {from_date} to present")
            print(f"Fetching data from {from_date} to present")
            
            # Force refresh to avoid cache issues (use our fixed pagination)
            market_data = self.data_connector.fetch_market_data(
                symbol=symbol, 
                timeframe=timeframe, 
                from_date=from_date,
                force_refresh=force_refresh  # Force refresh to avoid cache issues
            )
            
            if market_data.empty:
                error_msg = f"Could not fetch market data for {symbol}"
                logger.error(error_msg)
                print(f"Error: {error_msg}")
                return 1
            
            logger.info(f"Retrieved {len(market_data)} candles from {market_data.index[0]} to {market_data.index[-1]}")
            print(f"Retrieved {len(market_data)} candles from {market_data.index[0].strftime('%Y-%m-%d')} to {market_data.index[-1].strftime('%Y-%m-%d')}")
            
            # 2. Generate technical features directly (skip caching to avoid column mismatch)
            logger.debug(f"Generating technical indicators for {symbol}")
            
            # Manually calculate indicators to avoid cache issues
            df_indicators = self.technical_features._calculate_indicators(market_data)
            
            print(f"Generated technical indicators. Data shape: {df_indicators.shape}")
            print(f"Indicators: {', '.join([col for col in df_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']])}")
            
            # 3. Run advanced models (PCA and HMM)
            logger.debug(f"Running advanced statistical models for {symbol}")
            print("\nRunning PCA and market regime detection...")
            
            # Run all models
            df_with_regimes, stats = run_models(
                df=df_indicators,
                output_dir=output_dir,
                n_components=n_components,
                n_regimes=n_regimes,
                create_plots=True
            )
            
            # Extract and display PCA results
            pca_variance = stats['pca']['variance']['explained_variance_ratio']
            pca_cumulative = stats['pca']['variance']['cumulative_explained_variance']
            
            print("\nPCA Results:")
            print(f"Components: {n_components}")
            for i, var in enumerate(pca_variance):
                print(f"  PC{i+1} explains {var*100:.2f}% of variance")
            print(f"  Total explained variance: {pca_cumulative[-1]*100:.2f}%")
            
            # Display current regime
            current_kmeans_regime = df_with_regimes['kmeans_regime'].iloc[-1]
            current_hmm_regime = df_with_regimes['hmm_regime'].iloc[-1]
            
            print("\nCurrent Market Regimes:")
            print(f"  KMeans cluster: Regime {current_kmeans_regime}")
            print(f"  HMM state: Regime {current_hmm_regime}")
            
            # Describe regimes
            print("\nRegime Characteristics (KMeans):")
            for regime, data in stats['kmeans_regimes'].items():
                print(f"  Regime {regime}:")
                print(f"    Frequency: {data['count']} days ({data['count']/len(df_with_regimes)*100:.1f}%)")
                print(f"    Avg price: ${data['avg_close']:.2f}")
                print(f"    Avg return: {data['avg_return']:.2f}%")
                print(f"    Volatility: {data['volatility']:.2f}%")
                print(f"    Win rate: {data['win_rate']:.1f}%")
                print(f"    Avg duration: {data['avg_duration']:.1f} days")
            
            # Save important data to CSV
            final_df = df_with_regimes[['open', 'high', 'low', 'close', 'volume', 
                                      'PC1', 'PC2', 'kmeans_regime', 'hmm_regime']]
            csv_path = f"{output_dir}/{symbol.replace('/', '_')}_{timeframe}_regimes.csv"
            final_df.to_csv(csv_path)
            
            # Save stats as JSON
            json_stats = {k: v for k, v in stats.items() if k != 'pca'}  # Skip PCA details in JSON (too complex)
            json_path = f"{output_dir}/{symbol.replace('/', '_')}_{timeframe}_stats.json"
            with open(json_path, 'w') as f:
                json.dump(json_stats, f, indent=2, default=str)
            
            print(f"\nAnalysis complete. Results saved to {output_dir}")
            print(f"CSV data: {csv_path}")
            print(f"JSON stats: {json_path}")
            
            # 6. Generate market summary with LLM if available
            if os.environ.get("LLM_API_KEY"):
                logger.debug(f"Generating LLM market analysis for advanced models")
                print("\nGenerating advanced analysis summary with LLM...")
                
                # Create a prompt for the LLM with advanced analysis results
                prompt = f"""
                Analyze the following market data and regime detection results for {symbol}:
                
                Current price: ${df_with_regimes['close'].iloc[-1]:.2f}
                
                PCA Results:
                - {n_components} principal components explain {pca_cumulative[-1]*100:.2f}% of variance
                
                Current Market Regimes:
                - KMeans cluster: Regime {current_kmeans_regime}
                - HMM state: Regime {current_hmm_regime}
                
                Regime {current_kmeans_regime} characteristics:
                - Frequency: {stats['kmeans_regimes'][current_kmeans_regime]['count']} days ({stats['kmeans_regimes'][current_kmeans_regime]['count']/len(df_with_regimes)*100:.1f}%)
                - Avg return: {stats['kmeans_regimes'][current_kmeans_regime]['avg_return']:.2f}%
                - Volatility: {stats['kmeans_regimes'][current_kmeans_regime]['volatility']:.2f}%
                - Win rate: {stats['kmeans_regimes'][current_kmeans_regime]['win_rate']:.1f}%
                
                Provide a concise market analysis based on this information. Include interpretation of what these regimes typically indicate and potential scenarios that might unfold.
                """
                
                summary = self.llm.analyze_with_llm(prompt)
                logger.info(f"LLM advanced analysis generated for {symbol}")
                print("\nLLM Analysis of Advanced Models:")
                print(summary)
            else:
                logger.warning("LLM analysis skipped (no API key provided)")
                print("\nLLM Analysis skipped (no API key provided)")
            
        logger.info(f"Advanced analysis of {symbol} completed successfully")
        return 0
    
    def list_symbols(self):
        """List available trading pairs"""
        logger.info("Listing available trading symbols")
        with ErrorHandler(context="fetching symbol list") as handler:
            exchange = self.data_connector.exchange
            markets = exchange.load_markets()
            symbols = list(markets.keys())
            
            logger.info(f"Retrieved {len(symbols)} symbols from {exchange.name}")
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
                    logger.debug(f"Displaying {currency} pairs ({len(pairs)} total)")
                    print(f"  {currency}: {', '.join(pairs[:5])}{'...' if len(pairs) > 5 else ''}")
                    del by_base[currency]
            
            # Print remaining currencies
            for currency, pairs in list(by_base.items())[:10]:  # Limit to avoid too much output
                print(f"  {currency}: {', '.join(pairs[:5])}{'...' if len(pairs) > 5 else ''}")
            
            if len(by_base) > 10:
                logger.debug(f"Omitted {len(by_base) - 10} base currencies from display")
                print(f"  ... and {len(by_base) - 10} more base currencies")
            
            logger.info("Symbol list displayed successfully")
            return 0
    
    def list_timeframes(self):
        """List available timeframes"""
        logger.info("Listing available timeframes")
        with ErrorHandler(context="fetching timeframes") as handler:
            exchange = self.data_connector.exchange
            timeframes = exchange.timeframes
            
            logger.info(f"Retrieved {len(timeframes)} timeframes from {exchange.name}")
            print(f"Available timeframes on {exchange.name}:")
            for tf, description in timeframes.items():
                print(f"  {tf}: {description}")
            
            logger.info("Timeframes displayed successfully")
            return 0
    
    def backtest(self, conditions: List[str], symbol: str, timeframe: str, days: int):
        """Backtest specific market conditions"""
        condition_str = ", ".join(conditions)
        print(f"Backtesting conditions: {condition_str}")
        print(f"Symbol: {symbol}, Timeframe: {timeframe}, Data period: {days} days\n")
        
        logger.info(f"Starting backtest for conditions [{condition_str}] on {symbol} ({timeframe}, {days} days)")
        
        with ErrorHandler(context=f"backtesting {condition_str} on {symbol}") as handler:
            # 1. Fetch data
            logger.debug(f"Fetching {days} days of data for {symbol}")
            market_data = self.data_connector.fetch_market_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=days
            )
            
            if market_data.empty:
                error_msg = f"Could not fetch market data for {symbol}"
                logger.error(error_msg)
                print(f"Error: {error_msg}")
                return 1
            
            logger.info(f"Retrieved {len(market_data)} candles for {symbol}")
            
            # 2. Generate technical features
            logger.debug(f"Generating technical indicators for {symbol}")
            df_indicators = self.technical_features.add_indicators(market_data, symbol, timeframe)
            
            # 3. Find similar dates
            logger.debug(f"Finding instances of conditions: {condition_str}")
            similar_dates = self.backtester.find_similar_conditions(df_indicators, conditions)
            
            if not similar_dates:
                logger.info(f"No instances of conditions [{condition_str}] found in historical data")
                print("No instances of these conditions found in the historical data.")
                return 0
            
            logger.info(f"Found {len(similar_dates)} instances of conditions [{condition_str}]")
            print(f"Found {len(similar_dates)} instances of specified conditions")
            
            # 4. Calculate forward returns
            logger.debug(f"Calculating returns for {len(similar_dates)} similar instances")
            results_df, stats = self.backtester.analyze_forward_returns(df_indicators, similar_dates)
            
            print("\nHistorical Forward Returns:")
            for period, value in stats.items():
                if 'mean' in period:
                    days = period.split('d_')[0]
                    logger.info(f"{days}-day forward return (avg): {value:.2f}%")
                    print(f"  {days}-day forward return (avg): {value:.2f}%")
                if 'positive_pct' in period:
                    days = period.split('d_')[0]
                    logger.info(f"{days}-day win rate: {value:.2f}%")
                    print(f"  {days}-day win rate: {value:.2f}%")
            
            if not results_df.empty:
                logger.debug("Displaying most recent occurrences")
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
            
            logger.info(f"Backtest for conditions [{condition_str}] completed successfully")
            return 0
            
    def fetch_history(self, symbol: str, timeframe: str, from_date: str, to_date: str = None, 
                       output: str = None, add_indicators: bool = False):
        """Fetch complete historical data
        
        Args:
            symbol: Trading pair to fetch
            timeframe: Timeframe to fetch
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format (defaults to current date)
            output: Output file path for CSV export
            add_indicators: Whether to calculate and cache technical indicators
        """
        print(f"Fetching complete historical data for {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Period: {from_date} to {to_date or 'now'}")
        if add_indicators:
            print("Technical indicators will be calculated\n")
        else:
            print()
        
        logger.info(f"Starting historical data fetch for {symbol} ({timeframe}) from {from_date} to {to_date or 'now'}")
        
        with ErrorHandler(context=f"fetching historical data for {symbol}") as handler:
            # Handle output paths based on type of data being saved
            formatted_symbol = symbol.replace('/', '_')
            
            # Determine OHLCV output path
            ohlcv_output = output
            if output is None:
                ohlcv_output = f"{self.data_connector.cache.cache_dir}/{formatted_symbol}_{timeframe}_ohlcv.csv"
                logger.info(f"Using default output filename: {ohlcv_output}")
                
            # Determine path for indicators file (for informational purposes)
            indicator_path = f"{self.data_connector.cache.indicators_dir}/{formatted_symbol}_{timeframe}_indicators.csv"
            
            # Fetch data with the unified function
            market_data = self.data_connector.fetch_market_data(
                symbol=symbol,
                timeframe=timeframe,
                from_date=from_date,
                to_date=to_date,
                add_indicators=add_indicators,
                csv_output=ohlcv_output if not add_indicators else None  # Only pass output path for OHLCV data
            )
            
            if market_data.empty:
                error_msg = f"Could not fetch historical data for {symbol}"
                logger.error(error_msg)
                print(f"Error: {error_msg}")
                return 1
            
            # Print data statistics
            print(f"Successfully retrieved {len(market_data)} candles")
            print(f"Time range: {market_data.index[0]} to {market_data.index[-1]}")
            
            if add_indicators:
                indicator_count = len(market_data.columns) - 5  # Subtract OHLCV columns
                print(f"Generated {indicator_count} technical indicators")
                print(f"\nIndicators saved to {indicator_path}")
            else:
                print(f"\nData successfully saved to {ohlcv_output}")
            
            # Print sample of the data
            print("\nData Sample (latest 5 candles):")
            pd.set_option('display.precision', 2)
            pd.set_option('display.max_columns', 10 if add_indicators else 5)  # Limit columns for readability
            pd.set_option('display.width', 120)
            print(market_data.tail(5).to_string())
            
            logger.info(f"Historical data fetch completed successfully for {symbol} ({len(market_data)} records)")
            return 0

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="Quant System CLI")
    
    # Add global debug flag that applies to all commands
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a market")
    analyze_parser.add_argument("--symbol", "-s", default="BTC/USDT", help="Trading pair to analyze")
    analyze_parser.add_argument("--timeframe", "-t", default="1d", help="Analysis timeframe")
    analyze_parser.add_argument("--days", "-d", type=int, default=365, help="Days of historical data")
    
    # Advanced analysis command
    advanced_parser = subparsers.add_parser("advanced", help="Run advanced statistical analysis (PCA, HMM)")
    advanced_parser.add_argument("--symbol", "-s", default="BTC/USDT", help="Trading pair to analyze")
    advanced_parser.add_argument("--timeframe", "-t", default="1d", help="Analysis timeframe") 
    advanced_parser.add_argument("--days", "-d", type=int, default=500, help="Days of historical data")
    advanced_parser.add_argument("--components", "-c", type=int, default=2, help="Number of PCA components")
    advanced_parser.add_argument("--regimes", "-r", type=int, default=3, help="Number of market regimes")
    advanced_parser.add_argument("--output", "-o", help="Output directory for analysis results")
    advanced_parser.add_argument("--force-refresh", "-f", action="store_true", help="Force refresh data from API")
    
    # List symbols command
    list_symbols_parser = subparsers.add_parser("symbols", help="List available trading pairs")
    
    # List timeframes command
    list_timeframes_parser = subparsers.add_parser("timeframes", help="List available timeframes")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Backtest market conditions")
    backtest_parser.add_argument("--conditions", "-c", required=True, nargs="+", help="Market conditions to backtest")
    backtest_parser.add_argument("--symbol", "-s", default="BTC/USDT", help="Trading pair to analyze")
    backtest_parser.add_argument("--timeframe", "-t", default="1d", help="Analysis timeframe")
    backtest_parser.add_argument("--days", "-d", type=int, default=365, help="Days of historical data")
    
    # Fetch historical data command
    history_parser = subparsers.add_parser("history", help="Fetch complete historical data")
    history_parser.add_argument("--symbol", "-s", default="BTC/USDT", help="Trading pair to fetch")
    history_parser.add_argument("--timeframe", "-t", default="1d", help="Data timeframe")
    history_parser.add_argument("--from", dest="from_date", required=True, help="Start date (YYYY-MM-DD)")
    history_parser.add_argument("--to", dest="to_date", help="End date (YYYY-MM-DD), defaults to now")
    history_parser.add_argument("--output", "-o", help="Output file path (CSV)")
    history_parser.add_argument("--indicators", "-i", action="store_true", help="Calculate and cache technical indicators")
    
    # Ensure we don't error on empty args
    if len(sys.argv) <= 1:
        parser.print_help()
        return 0
    
    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # If argument parsing fails, show help and exit with error code
        parser.print_help()
        return e.code
    
    # Set debug logging if requested
    if args.debug:
        set_log_level(DEBUG)
        logger.debug("Debug logging enabled in CLI")
    
    logger.info(f"Starting CLI with command: {args.command or 'help'}")
    
    try:
        cli = QuantSystemCLI()
        
        # Execute the command
        if args.command == "analyze":
            return cli.analyze(args.symbol, args.timeframe, args.days)
        elif args.command == "advanced":
            return cli.analyze_advanced(
                args.symbol, 
                args.timeframe, 
                args.days, 
                args.output, 
                args.components, 
                args.regimes,
                args.force_refresh
            )
        elif args.command == "symbols":
            return cli.list_symbols()
        elif args.command == "timeframes":
            return cli.list_timeframes()
        elif args.command == "backtest":
            return cli.backtest(args.conditions, args.symbol, args.timeframe, args.days)
        elif args.command == "history":
            return cli.fetch_history(args.symbol, args.timeframe, args.from_date, args.to_date, args.output, args.indicators)
        else:
            logger.info("No command specified, showing help")
            parser.print_help()
            return 0
    except Exception as e:
        logger.critical(f"Unhandled exception in CLI: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())