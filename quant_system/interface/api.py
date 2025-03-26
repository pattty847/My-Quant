# interface/api.py
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import time
import traceback

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our system components
from quant_system.data.connectors import CryptoDataConnector
from quant_system.features.technical import TechnicalFeatures
from quant_system.analysis.market_structure import MarketStructureAnalyzer
from quant_system.analysis.backtest import MarketBacktester
from quant_system.analysis.llm_interface import LLMAnalyzer
from quant_system.utils import get_logger, ErrorHandler

# Initialize logger for this module
logger = get_logger("interface.api")

app = FastAPI(
    title="Quant System API",
    description="API for interacting with the Quantitative Trading System",
    version="0.1.0"
)

# Define request and response models
class AnalysisRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1d"
    days: int = 365
    include_raw_data: bool = False

class AnalysisResponse(BaseModel):
    symbol: str
    conditions: List[str]
    similar_instances: int
    backtest_stats: Dict[str, float]
    summary: str
    raw_data: Optional[Dict[str, Any]] = None

# Initialize system components
logger.info("Initializing system components")
try:
    data_connector = CryptoDataConnector()
    technical_features = TechnicalFeatures()
    market_analyzer = MarketStructureAnalyzer()
    backtester = MarketBacktester()
    llm_analyzer = LLMAnalyzer(api_key=os.environ.get("LLM_API_KEY"))
    logger.info("System components initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize system components: {e}")
    logger.debug(traceback.format_exc())
    raise

@app.get("/")
def read_root():
    logger.debug("Root endpoint accessed")
    return {"message": "Welcome to the Quant System API"}

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_market(request: AnalysisRequest):
    """Run a complete market analysis for a given symbol"""
    request_id = f"req_{int(time.time())}"
    logger.info(f"[{request_id}] Market analysis requested for {request.symbol} on {request.timeframe} timeframe ({request.days} days)")
    
    with ErrorHandler(context=f"market analysis for {request.symbol}") as handler:
        # 1. Fetch data
        logger.debug(f"[{request_id}] Fetching market data")
        market_data = data_connector.fetch_ohlcv(
            symbol=request.symbol, 
            timeframe=request.timeframe, 
            limit=request.days
        )
        
        if market_data.empty:
            logger.warning(f"[{request_id}] No market data found for {request.symbol}")
            raise HTTPException(status_code=404, detail="Could not fetch market data")
        
        logger.debug(f"[{request_id}] Retrieved {len(market_data)} candles from {market_data.index[0]} to {market_data.index[-1]}")
        
        # 2. Generate technical features
        logger.debug(f"[{request_id}] Generating technical indicators")
        df_indicators = technical_features.add_indicators(market_data)
        
        # 3. Identify current market conditions
        logger.debug(f"[{request_id}] Identifying market conditions")
        conditions = market_analyzer.identify_conditions(df_indicators)
        logger.info(f"[{request_id}] Identified conditions: {', '.join(conditions)}")
        
        # 4. Find similar historical conditions and analyze performance
        logger.debug(f"[{request_id}] Finding similar historical conditions")
        similar_dates = backtester.find_similar_conditions(df_indicators, conditions)
        logger.info(f"[{request_id}] Found {len(similar_dates)} similar historical instances")
        
        # 5. Calculate forward returns from similar conditions
        logger.debug(f"[{request_id}] Calculating historical forward returns")
        results_df, stats = backtester.analyze_forward_returns(df_indicators, similar_dates)
        
        # 6. Generate market summary with LLM
        logger.debug(f"[{request_id}] Generating market summary with LLM")
        summary = llm_analyzer.generate_market_summary(df_indicators, conditions, stats)
        
        # 7. Prepare response
        logger.debug(f"[{request_id}] Preparing API response")
        response = {
            "symbol": request.symbol,
            "conditions": conditions,
            "similar_instances": len(similar_dates),
            "backtest_stats": stats,
            "summary": summary,
            "raw_data": None
        }
        
        # Include raw data if requested
        if request.include_raw_data:
            logger.debug(f"[{request_id}] Including raw data in response")
            response["raw_data"] = {
                "recent_data": df_indicators.tail(20).to_dict(),
                "similar_instances": results_df.to_dict() if not results_df.empty else {}
            }
        
        logger.info(f"[{request_id}] Analysis completed successfully")
        return response

@app.get("/symbols")
def get_available_symbols():
    """Get available trading pairs from the exchange"""
    logger.info("Symbol list requested")
    with ErrorHandler(context="fetching symbols list") as handler:
        exchange = data_connector.exchange
        markets = exchange.load_markets()
        symbols = list(markets.keys())
        logger.info(f"Retrieved {len(symbols)} symbols")
        return {"symbols": symbols}

@app.get("/timeframes")
def get_available_timeframes():
    """Get available timeframes from the exchange"""
    logger.info("Timeframes list requested")
    with ErrorHandler(context="fetching timeframes") as handler:
        exchange = data_connector.exchange
        timeframes = exchange.timeframes
        logger.info(f"Retrieved {len(timeframes)} timeframes")
        return {"timeframes": timeframes}

@app.get("/conditions")
def get_current_conditions(
    symbol: str = Query("BTC/USDT", description="Trading pair symbol"),
    timeframe: str = Query("1d", description="Timeframe for analysis")
):
    """Get current market conditions for a symbol"""
    logger.info(f"Current conditions requested for {symbol} on {timeframe} timeframe")
    with ErrorHandler(context=f"analyzing conditions for {symbol}") as handler:
        # Fetch last 50 days of data
        logger.debug(f"Fetching data for {symbol}")
        market_data = data_connector.fetch_ohlcv(symbol, timeframe, limit=50)
        
        if market_data.empty:
            logger.warning(f"No market data found for {symbol}")
            raise HTTPException(status_code=404, detail="Could not fetch market data")
        
        # Generate indicators
        logger.debug(f"Generating indicators for {symbol}")
        df_indicators = technical_features.add_indicators(market_data)
        
        # Identify conditions
        logger.debug(f"Identifying conditions for {symbol}")
        conditions = market_analyzer.identify_conditions(df_indicators)
        logger.info(f"Identified conditions for {symbol}: {', '.join(conditions)}")
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "conditions": conditions,
            "last_price": float(market_data.iloc[-1]['close']),
            "last_update": market_data.index[-1].isoformat()
        }

@app.get("/backtest")
def backtest_condition(
    condition: List[str] = Query(..., description="Market conditions to backtest"),
    symbol: str = Query("BTC/USDT", description="Trading pair symbol"),
    timeframe: str = Query("1d", description="Timeframe for analysis"),
    days: int = Query(365, description="Days of historical data to analyze")
):
    """Backtest specific market conditions"""
    condition_str = ", ".join(condition)
    logger.info(f"Backtest requested for conditions [{condition_str}] on {symbol} ({timeframe}, {days} days)")
    
    with ErrorHandler(context=f"backtesting {condition_str} on {symbol}") as handler:
        # Fetch data
        logger.debug(f"Fetching {days} days of data for {symbol}")
        market_data = data_connector.fetch_ohlcv(symbol, timeframe, limit=days)
        
        if market_data.empty:
            logger.warning(f"No market data found for {symbol}")
            raise HTTPException(status_code=404, detail="Could not fetch market data")
        
        # Generate indicators
        logger.debug(f"Generating indicators for {symbol}")
        df_indicators = technical_features.add_indicators(market_data)
        
        # Find similar dates
        logger.debug(f"Finding instances of conditions: {condition_str}")
        similar_dates = backtester.find_similar_conditions(df_indicators, condition)
        
        if not similar_dates:
            logger.info(f"No similar conditions found for {condition_str}")
            return {
                "symbol": symbol,
                "conditions": condition,
                "message": "No similar conditions found in the historical data"
            }
        
        # Calculate forward returns
        logger.debug(f"Calculating returns for {len(similar_dates)} similar instances")
        results_df, stats = backtester.analyze_forward_returns(df_indicators, similar_dates)
        
        # Prepare simplified results
        simplified_results = []
        if not results_df.empty:
            for _, row in results_df.iterrows():
                simplified_results.append({
                    "date": row["date"].isoformat(),
                    "conditions": row["conditions"],
                    "returns": {
                        k: float(v) for k, v in row.items() 
                        if k not in ["date", "conditions"] and not pd.isna(v)
                    }
                })
        
        logger.info(f"Backtest completed with {len(similar_dates)} instances")
        return {
            "symbol": symbol,
            "conditions": condition,
            "instances": len(similar_dates),
            "stats": stats,
            "examples": simplified_results[:10]  # Limit to first 10 examples
        }

# Run with: uvicorn interface.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)