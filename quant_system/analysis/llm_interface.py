import json
import requests
import os

class LLMAnalyzer:
    """Interface with LLMs for market analysis"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        
    def generate_market_summary(self, market_data, conditions, backtest_stats):
        """Generate a market summary using an LLM"""
        # For simplicity, we'll use a template for now
        # In production, this would call an LLM API
        
        # Create a simplified market summary template
        recent_price = market_data.iloc[-1]['close']
        price_change_1d = (recent_price - market_data.iloc[-2]['close']) / market_data.iloc[-2]['close'] * 100
        price_change_7d = (recent_price - market_data.iloc[-7]['close']) / market_data.iloc[-7]['close'] * 100
        
        summary = f"""
        Market Summary:
        - Current price: ${recent_price:.2f}
        - 24h change: {price_change_1d:.2f}%
        - 7d change: {price_change_7d:.2f}%

        Market Conditions:
        - {', '.join(conditions)}

        Historical Performance (similar conditions):
        """
        
        # Add backtest stats
        for period, value in backtest_stats.items():
            if 'mean' in period:
                days = period.split('d_')[0]
                summary += f"- {days}-day forward return (avg): {value:.2f}%\n"
            if 'positive_pct' in period:
                days = period.split('d_')[0]
                summary += f"- {days}-day win rate: {value:.2f}%\n"
        
        return summary
    
    def analyze_with_llm(self, prompt, max_tokens=500):
        """Send a prompt to LLM API and get a response"""
        # This is a placeholder for actual LLM API call
        # In a real implementation, you would call Claude, GPT-4, etc.
        
        if not self.api_key:
            return "No API key provided for LLM service"
        
        # Example implementation for Claude API
        # Would need to be adapted based on the specific LLM service used
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                return f"Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"Error calling LLM API: {str(e)}"
