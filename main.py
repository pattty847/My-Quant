# main.py
import os
import sys
import argparse
import uvicorn
from dotenv import load_dotenv
from quant_system.utils import logger, set_log_level, DEBUG, INFO, ErrorHandler

# Load environment variables from .env file
logger.info("Loading environment variables from .env file")
load_dotenv()

def start_api(host="0.0.0.0", port=8000, reload=False, debug=False):
    """Start the FastAPI server"""
    if debug:
        set_log_level(DEBUG)
        logger.debug("Debug logging enabled for API")
    
    logger.info(f"Starting API server on {host}:{port} (reload={reload})")
    try:
        uvicorn.run("quant_system.interface.api:app", host=host, port=port, reload=reload)
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise

def start_cli():
    """Start the command-line interface"""
    logger.info("Starting CLI interface")
    try:
        # Import and run the CLI directly, letting it handle its own arguments
        from quant_system.interface.cli import main
        main()
    except Exception as e:
        logger.error(f"Failed to start CLI interface: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant System")
    
    # Top level command
    subparsers = parser.add_subparsers(dest="mode", help="Run mode")
    
    # API mode
    api_parser = subparsers.add_parser("api", help="Start the web API")
    api_parser.add_argument("--host", default="0.0.0.0", help="API host")
    api_parser.add_argument("--port", "-p", type=int, default=8000, help="API port")
    api_parser.add_argument("--reload", "-r", action="store_true", help="Enable auto-reload")
    api_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # CLI mode - we'll pass all args to the CLI parser
    cli_parser = subparsers.add_parser("cli", help="Run command-line interface")
    
    # If no args or just help flag, show help
    if len(sys.argv) <= 1 or (len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']):
        parser.print_help()
        sys.exit(0)
    
    # Handle the case where we want to run the CLI with arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'cli':
        # Remove 'main.py' and 'cli' from the args, as we're going to pass control
        # directly to the CLI module
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        start_cli()
        sys.exit(0)
    
    # For API mode, parse arguments normally
    args = parser.parse_args()
    
    logger.info(f"Starting Quant System in {args.mode} mode")
    
    with ErrorHandler(context="main execution", exit_on_error=True):
        if args.mode == "api":
            start_api(args.host, args.port, args.reload, args.debug)
        else:
            parser.print_help()
            sys.exit(1)