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

def start_api(host="0.0.0.0", port=8000, reload=False):
    """Start the FastAPI server"""
    logger.info(f"Starting API server on {host}:{port} (reload={reload})")
    try:
        uvicorn.run("quant_system.interface.api:app", host=host, port=port, reload=reload)
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise

def start_cli(debug=False):
    """Start the command-line interface"""
    logger.info("Starting CLI interface")
    try:
        from quant_system.interface.cli import main
        
        # If debug mode is enabled in the main script, ensure it's passed to CLI
        if debug and '--debug' not in sys.argv:
            sys.argv.append('--debug')
            
        main()
    except Exception as e:
        logger.error(f"Failed to start CLI interface: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant System")
    subparsers = parser.add_subparsers(dest="mode", help="Run mode")
    
    # API mode
    api_parser = subparsers.add_parser("api", help="Start the web API")
    api_parser.add_argument("--host", default="0.0.0.0", help="API host")
    api_parser.add_argument("--port", "-p", type=int, default=8000, help="API port")
    api_parser.add_argument("--reload", "-r", action="store_true", help="Enable auto-reload")
    
    # CLI mode - forward all remaining arguments to the CLI
    cli_parser = subparsers.add_parser("cli", help="Run command-line interface")
    
    # Add common arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args, remaining = parser.parse_known_args()
    
    # Set logging level based on arguments
    if hasattr(args, 'debug') and args.debug:
        set_log_level(DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info(f"Starting Quant System in {args.mode or 'help'} mode")
    
    with ErrorHandler(context="main execution", exit_on_error=True):
        if args.mode == "api":
            start_api(args.host, args.port, args.reload)
        elif args.mode == "cli":
            # Pass remaining arguments to CLI
            sys.argv = [sys.argv[0]] + remaining
            start_cli(debug=args.debug)
        else:
            logger.info("No mode specified, showing help")
            parser.print_help()