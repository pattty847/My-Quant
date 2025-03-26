#!/bin/bash
echo "========================================"
echo "   Quant System - Debug Examples"
echo "========================================"
echo

echo "### API Server with Debug Logging ###"
echo "python main.py api --debug"
echo

echo "### CLI Analyze Command with Debug Logging ###"
echo "python main.py cli analyze --symbol BTC/USDT --timeframe 1d --days 100 --debug"
echo

echo "### CLI Symbols Command with Debug Logging ###"
echo "python main.py cli symbols --debug"
echo

echo "### CLI Backtest Command with Debug Logging ###"
echo "python main.py cli backtest --conditions \"RSI below 30\" \"MACD bullish crossover\" --symbol ETH/USDT --debug"
echo

echo "### Running Commands Directly ###"
echo "python -m quant_system.interface.cli analyze --symbol BTC/USDT --timeframe 1d --days 100 --debug"
echo

echo "Note: All logs are also saved to the logs/ directory with timestamps"
echo 