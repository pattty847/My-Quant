# interface/api.py
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our system components
from quant_system.data.connectors import CryptoDataConnector
from quant_system.features.technical import TechnicalFeatures
from quant_system.analysis.market_structure import MarketStructureAnalyzer
from quant_system.analysis.backtest import MarketBacktester
from quant_system.analysis.llm_interface import LLMAnalyzer

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
data_connector = CryptoDataConnector()
technical_features = TechnicalFeatures()
market_analyzer = MarketStructureAnalyzer()
backtester = MarketBacktester()
llm_analyzer = LLMAnalyzer(api_key=os.environ.get("LLM_API_KEY"))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Quant System API"}

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_market(request: AnalysisRequest):
    """Run a complete market analysis for a given symbol"""
    try:
        # 1. Fetch data
        market_data = data_connector.fetch_ohlcv(
            symbol=request.symbol, 
            timeframe=request.timeframe, 
            limit=request.days
        )
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail="Could not fetch market data")
        
        # 2. Generate technical features
        df_indicators = technical_features.add_indicators(market_data)
        
        # 3. Identify current market conditions
        conditions = market_analyzer.identify_conditions(df_indicators)
        
        # 4. Find similar historical conditions and analyze performance
        similar_dates = backtester.find_similar_conditions(df_indicators, conditions)
        
        # 5. Calculate forward returns from similar conditions
        results_df, stats = backtester.analyze_forward_returns(df_indicators, similar_dates)
        
        # 6. Generate market summary with LLM
        summary = llm_analyzer.generate_market_summary(df_indicators, conditions, stats)
        
        # 7. Prepare response
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
            response["raw_data"] = {
                "recent_data": df_indicators.tail(20).to_dict(),
                "similar_instances": results_df.to_dict() if not results_df.empty else {}
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/symbols")
def get_available_symbols():
    """Get available trading pairs from the exchange"""
    try:
        exchange = data_connector.exchange
        markets = exchange.load_markets()
        symbols = list(markets.keys())
        return {"symbols": symbols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching symbols: {str(e)}")

@app.get("/timeframes")
def get_available_timeframes():
    """Get available timeframes from the exchange"""
    try:
        exchange = data_connector.exchange
        timeframes = exchange.timeframes
        return {"timeframes": timeframes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching timeframes: {str(e)}")

@app.get("/conditions")
def get_current_conditions(
    symbol: str = Query("BTC/USDT", description="Trading pair symbol"),
    timeframe: str = Query("1d", description="Timeframe for analysis")
):
    """Get current market conditions for a symbol"""
    try:
        # Fetch last 50 days of data
        market_data = data_connector.fetch_ohlcv(symbol, timeframe, limit=50)
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail="Could not fetch market data")
        
        # Generate indicators
        df_indicators = technical_features.add_indicators(market_data)
        
        # Identify conditions
        conditions = market_analyzer.identify_conditions(df_indicators)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "conditions": conditions,
            "last_price": float(market_data.iloc[-1]['close']),
            "last_update": market_data.index[-1].isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing conditions: {str(e)}")

@app.get("/backtest")
def backtest_condition(
    condition: List[str] = Query(..., description="Market conditions to backtest"),
    symbol: str = Query("BTC/USDT", description="Trading pair symbol"),
    timeframe: str = Query("1d", description="Timeframe for analysis"),
    days: int = Query(365, description="Days of historical data to analyze")
):
    """Backtest specific market conditions"""
    try:
        # Fetch data
        market_data = data_connector.fetch_ohlcv(symbol, timeframe, limit=days)
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail="Could not fetch market data")
        
        # Generate indicators
        df_indicators = technical_features.add_indicators(market_data)
        
        # Find similar dates
        similar_dates = backtester.find_similar_conditions(df_indicators, condition)
        
        if not similar_dates:
            return {
                "symbol": symbol,
                "conditions": condition,
                "message": "No similar conditions found in the historical data"
            }
        
        # Calculate forward returns
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
        
        return {
            "symbol": symbol,
            "conditions": condition,
            "instances": len(similar_dates),
            "stats": stats,
            "results": simplified_results[:10]  # Limit to 10 results for brevity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")

# Run with: uvicorn interface.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)