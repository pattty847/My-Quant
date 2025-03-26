#!/usr/bin/env python
"""
Logging System Demonstration Script

This script demonstrates the various logging features available in the Quant System.
Run it to see examples of different log levels, error handling, and best practices.
"""
import os
import sys
import time
import random

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our logging utilities
from quant_system.utils import (
    get_logger, 
    ErrorHandler, 
    set_log_level, 
    DEBUG, INFO, WARNING, ERROR, CRITICAL,
    logger as root_logger
)

# Create a module-specific logger
logger = get_logger("examples.logging_demo")

def demonstrate_log_levels():
    """Demonstrate different logging levels"""
    print("\n=== Logging Levels Demonstration ===")
    
    logger.debug("This is a DEBUG message - detailed information for troubleshooting")
    logger.info("This is an INFO message - confirmation that things are working as expected")
    logger.warning("This is a WARNING message - indication that something unexpected happened")
    logger.error("This is an ERROR message - the software has not been able to perform some function")
    logger.critical("This is a CRITICAL message - a serious error that might prevent program execution")
    
    # Show how to include context in log messages
    symbol = "BTC/USDT"
    price = 45000.00
    change = 2.5
    logger.info(f"Market update for {symbol}: Price ${price:.2f} (Change: {change:.2f}%)")

def demonstrate_error_handling():
    """Demonstrate error handling with the ErrorHandler context manager"""
    print("\n=== Error Handling Demonstration ===")
    
    # Example 1: Basic error handling (suppresses the exception)
    try:
        with ErrorHandler(context="division operation"):
            # Simulate an error
            result = 100 / 0
            print("This will not be executed")
    except Exception as e:
        print("This exception should not be caught, as ErrorHandler suppresses it")
    
    logger.info("Execution continues after the error")
    
    # Example 2: Error handling with exit_on_error=True
    # This would normally exit the program, so we'll just show the concept
    print("\n--- Error with exit_on_error=True (simulated) ---")
    logger.info("If exit_on_error=True was used, program would exit here")
    
    # Example 3: Handling different types of errors
    with ErrorHandler(context="data processing"):
        try:
            data = {"key": "value"}
            print(f"Processing data: {data['nonexistent_key']}")
        except KeyError:
            logger.warning("Handled a KeyError inside the ErrorHandler context")
    
    logger.info("Execution continues")

def simulate_market_analysis():
    """Simulate a market analysis with detailed logging"""
    print("\n=== Market Analysis Simulation ===")
    
    # Log the start of the operation
    symbol = "ETH/USDT"
    timeframe = "4h"
    days = 30
    logger.info(f"Starting analysis of {symbol} on {timeframe} timeframe ({days} days)")
    
    with ErrorHandler(context=f"market analysis for {symbol}"):
        # Simulate data fetching
        logger.debug(f"Fetching market data for {symbol}")
        time.sleep(0.5)  # Simulate API call
        
        # Simulate random failure (20% chance)
        if random.random() < 0.2:
            raise ConnectionError(f"Failed to connect to exchange API for {symbol}")
        
        # Log successful data retrieval
        candles = random.randint(80, 120)
        start_date = "2023-01-01"
        end_date = "2023-01-31"
        logger.info(f"Retrieved {candles} candles from {start_date} to {end_date}")
        
        # Simulate data processing
        logger.debug(f"Generating technical indicators for {symbol}")
        time.sleep(0.3)  # Simulate processing
        
        # Simulate identifying market conditions
        logger.debug(f"Identifying market conditions for {symbol}")
        conditions = ["Bullish trend", "Oversold RSI", "Above 200 SMA"]
        logger.info(f"Identified conditions for {symbol}: {', '.join(conditions)}")
        
        # Simulate calculating results
        similar_instances = random.randint(5, 15)
        logger.info(f"Found {similar_instances} similar historical instances for {symbol}")
        
        # Simulate generating results
        logger.debug(f"Calculating forward returns for {symbol}")
        forward_return = random.uniform(-5.0, 15.0)
        win_rate = random.uniform(40.0, 80.0)
        logger.info(f"10-day forward return (avg): {forward_return:.2f}%")
        logger.info(f"10-day win rate: {win_rate:.2f}%")
    
    logger.info(f"Analysis of {symbol} completed successfully")

def demonstrate_debug_logging():
    """Demonstrate the difference between INFO and DEBUG logging"""
    print("\n=== Debug Logging Demonstration ===")
    print("Current log level: INFO (default)")
    
    logger.debug("This DEBUG message will NOT appear in the console")
    logger.info("This INFO message will appear in the console")
    
    print("\nChanging log level to DEBUG...")
    set_log_level(DEBUG)
    
    logger.debug("This DEBUG message will NOW appear in the console")
    logger.info("This INFO message will still appear in the console")
    
    # Reset to INFO level
    set_log_level(INFO)

def main():
    """Main entry point for the logging demonstration"""
    print("=== Quant System Logging Demonstration ===")
    print("This script shows how the logging system works.")
    print("Note: All logs are also saved to the logs/ directory.")
    
    # Demonstrate basic log levels
    demonstrate_log_levels()
    
    # Demonstrate error handling
    demonstrate_error_handling()
    
    # Demonstrate a typical market analysis workflow
    simulate_market_analysis()
    
    # Demonstrate debug logging
    demonstrate_debug_logging()
    
    print("\n=== Logging Demonstration Complete ===")
    print(f"Check the logs directory for the complete log file.")

if __name__ == "__main__":
    main() 