# main.py
import os
import argparse
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def start_api(host="0.0.0.0", port=8000, reload=False):
    """Start the FastAPI server"""
    uvicorn.run("interface.api:app", host=host, port=port, reload=reload)

def start_cli():
    """Start the command-line interface"""
    from quant_system.interface.cli import main
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant System")
    parser.add_argument("--mode", "-m", choices=["api", "cli"], default="cli",
                        help="Run mode: api (web API) or cli (command-line)")
    parser.add_argument("--host", default="0.0.0.0", help="API host (only for API mode)")
    parser.add_argument("--port", "-p", type=int, default=8000, help="API port (only for API mode)")
    parser.add_argument("--reload", "-r", action="store_true", help="Enable auto-reload (only for API mode)")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        start_api(args.host, args.port, args.reload)
    else:
        start_cli()