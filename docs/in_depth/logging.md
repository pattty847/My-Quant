# Quant System Logging Guide

This document describes the logging system used throughout the Quant System project. Proper logging is essential for debugging, auditing, and understanding system behavior.

## Logging Architecture

The logging system uses Python's built-in `logging` module with a customized configuration. The system provides:

- Hierarchical loggers for different components
- Simultaneous console and file output
- Different logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Detailed error logging with tracebacks
- Error handling utilities
- Timestamp-based log files

## Log Files

Log files are stored in the `logs/` directory and named with a timestamp pattern: `quant_system_YYYYMMDD_HHMMSS.log`. Each run of the system creates a new log file.

## Log Levels

The system uses the following log levels:

- **DEBUG**: Detailed information for debugging and development
- **INFO**: General information about system operation
- **WARNING**: Indication of potential issues that don't prevent operation
- **ERROR**: Error conditions that may cause specific operations to fail
- **CRITICAL**: Critical errors that may cause the entire system to fail

Console output shows INFO level and above, while file logs capture DEBUG level and above.

## Running with Debug Logging

You can enable debug logging by adding the `--debug` flag to your commands:

### Using main.py

```bash
# API mode with debug logging
python main.py api --debug

# CLI analyze command with debug logging
python main.py cli analyze --symbol BTC/USDT --timeframe 1d --days 100 --debug

# CLI symbols command with debug logging
python main.py cli symbols --debug

# CLI backtest command with debug logging
python main.py cli backtest --conditions "RSI below 30" "MACD bullish crossover" --symbol ETH/USDT --debug
```

### Running CLI Commands Directly

You can also run the CLI commands directly:

```bash
# Run the CLI module directly with debug logging
python -m quant_system.interface.cli analyze --symbol BTC/USDT --timeframe 1d --days 100 --debug
```

## Using the Logging System

### Basic Logging

Import the logger and use it in your components:

```python
from quant_system.utils import get_logger

# Create a logger for your module
logger = get_logger("your.module.name")

# Log messages at different levels
logger.debug("Detailed information for debugging")
logger.info("General operational information")
logger.warning("Something unexpected but non-critical happened")
logger.error("An error occurred that impacts functionality")
logger.critical("A critical error occurred that may crash the system")
```

### Error Handling

Use the `ErrorHandler` context manager to handle and log exceptions:

```python
from quant_system.utils import ErrorHandler

def some_function():
    with ErrorHandler(context="descriptive operation name") as handler:
        # Your code here
        result = some_operation()
        
        # If an exception occurs within this block:
        # 1. It will be logged with context
        # 2. Full traceback will be saved to log file
        # 3. Exception will be suppressed (unless exit_on_error=True)
        
    return result
```

For critical operations that should exit on failure:

```python
with ErrorHandler(context="critical operation", exit_on_error=True) as handler:
    # If this fails, it will log and exit the program
    critical_operation()
```

### Setting Log Levels Programmatically

```python
from quant_system.utils import set_log_level, DEBUG, INFO, WARNING

# Set console output to debug level
set_log_level(DEBUG)

# Set console output to info level
set_log_level(INFO)
```

## Best Practices

1. **Create a module-specific logger**: Always use `get_logger("module.name")` rather than the root logger.

2. **Use appropriate log levels**: 
   - DEBUG for detailed information useful during development
   - INFO for operational information and important events
   - WARNING for unexpected but non-critical issues
   - ERROR for failures that impact specific operations
   - CRITICAL for system-wide failures

3. **Include context in log messages**: Log messages should include relevant identifiers and context.

4. **Use structured information**: For complex data, consider formatting as JSON or using key-value pairs.

5. **Use ErrorHandler for operations that may fail**: This ensures consistent error handling and logging.

6. **Don't log sensitive information**: Avoid logging API keys, credentials, or other sensitive data.

## Log Message Examples

Good log messages include context and clear descriptions:

- ✅ `logger.info(f"Analysis completed for {symbol} with {len(similar_dates)} similar instances found")`
- ✅ `logger.error(f"Failed to fetch data for {symbol}: {error_message}")`
- ✅ `logger.debug(f"Processing {len(data_points)} data points for technical indicator calculation")`

Bad log messages lack context or clarity:

- ❌ `logger.info("Analysis done")`
- ❌ `logger.error("Error occurred")`
- ❌ `logger.debug("Processing")`

## Viewing Log Files

To view the contents of log files, you can use standard tools:

### Windows

```
type logs\quant_system_*.log | more
```

### Linux/macOS

```
cat logs/quant_system_*.log | less
```

Or to watch logs in real-time:

```
tail -f logs/quant_system_*.log
``` 