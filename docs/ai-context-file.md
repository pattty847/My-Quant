# AI Assistant Context for Quant Analysis System

## Project Status (Last Updated: March 28, 2025)

### Current State
The quantitative analysis system is now fully functional with the following capabilities:
- Complete historical data fetching (1900+ candles) with proper pagination
- Technical indicator calculation (29 indicators) with on-demand processing
- PCA-based dimensionality reduction for feature analysis
- Market regime detection using both KMeans clustering and Hidden Markov Models
- Comprehensive visualizations (feature importance, regime characteristics, PC scatter plots)
- Results saved to timestamped folders with organized CSV, JSON, and visualization outputs

### Recent Fixes
- Fixed critical pagination bug in data connector that stopped after receiving fewer than 1000 candles
- Resolved column mismatch issues between cached indicators and newly calculated ones
- Implemented direct calculation of indicators for advanced analysis to avoid cache issues
- Enhanced CLI interface with `--force-refresh` flag and better progress reporting
- Improved error handling during data fetching and analysis processes
- Fixed visualization output to ensure all plots are properly saved

### System Components
- **Data Connectors**: Handle fetching from exchanges with proper pagination across date ranges
- **Cache System**: Stores OHLCV data and indicators (currently separate, planned for unification)
- **Technical Features**: Calculates 29 indicators with adaptive periods based on available data
- **Advanced Analysis**: PCA for dimensionality reduction, KMeans and HMM for regime detection
- **CLI Interface**: User commands for various analyses with consistent parameter handling
- **Visualization**: Generates multiple plot types saved to organized output directories

### Available Commands
- `analyze`: Basic market analysis with technical indicators and conditions
- `advanced`: Advanced analysis with PCA and regime detection (KMeans and HMM)
- `history`: Fetch historical data from specific date range and optionally calculate indicators
- `backtest`: Test specific market conditions against historical data
- `symbols`/`timeframes`: List available trading pairs and timeframes

### Current Focus
- Simplifying the cache structure to unify OHLCV and indicators in a single format
- Refining market regime detection for more accurate classification
- Enhancing visualization with interactive elements
- Improving error handling and recovery during long data fetches
- Considering a separate module for storing and analyzing detected regimes

### Implementation Notes
- The system separates data fetching, indicator calculation, and analysis modules
- There's some redundancy between history command and force-refresh flag
- Technical indicator calculation adapts to available data length
- Advanced analysis can handle missing values and partial data
- Cache mechanism needs simplification to avoid synchronization issues

### Envisioned Improvements
- Simpler cache structure with unified OHLCV + indicators format
- On-demand indicator calculation without caching complexity
- More sophisticated regime detection with transition probability analysis
- Integration of fundamental and macro data alongside technical indicators
- Machine learning models to predict regime transitions
- Better visualization with interactive charts (possibly Plotly implementation)

### File Organization
- `quant_system/data/`: Data fetching and caching (connectors.py, cache.py)
- `quant_system/features/`: Technical indicators (technical.py)
- `quant_system/analysis/`: Market analysis, advanced models, and backtesting
- `quant_system/interface/`: CLI and API interfaces
- `docs/`: Documentation files and AI context information
- `analysis_results/`: Output visualizations and statistics (organized by symbol and timestamp)
