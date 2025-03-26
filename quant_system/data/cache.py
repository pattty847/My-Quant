# quant_system/data/cache.py
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import json

from quant_system.utils import get_logger

# Initialize logger
logger = get_logger("data.cache")

class DataCache:
    """Cache for market data to reduce API calls and persist historical data"""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the data cache
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        
        # Create cache directories if they don't exist
        self.ohlcv_dir = self.cache_dir / "ohlcv"
        self.indicators_dir = self.cache_dir / "indicators"
        
        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)
        self.indicators_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized data cache in {self.cache_dir}")
        
        # Load cache metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file"""
        if not self.metadata_file.exists():
            logger.debug("No cache metadata file found, creating new metadata")
            return {"last_updated": {}, "cache_stats": {"hits": 0, "misses": 0}}
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                logger.debug("Loaded cache metadata")
                return metadata
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
            return {"last_updated": {}, "cache_stats": {"hits": 0, "misses": 0}}
    
    def _save_metadata(self):
        """Save cache metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                logger.debug("Saved cache metadata")
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_ohlcv_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get the path for a cached OHLCV file"""
        safe_symbol = symbol.replace('/', '_')
        return self.ohlcv_dir / f"{safe_symbol}_{timeframe}.csv"
    
    def _get_indicators_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get the path for a cached indicators file"""
        safe_symbol = symbol.replace('/', '_')
        return self.indicators_dir / f"{safe_symbol}_{timeframe}_indicators.csv"
    
    def get_cached_ohlcv(self, symbol: str, timeframe: str, max_age_days: Optional[int] = 1) -> Optional[pd.DataFrame]:
        """Retrieve cached OHLCV data if available and not too old
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '1d', '1h')
            max_age_days: Maximum age of cache in days before considering it stale
            
        Returns:
            DataFrame with OHLCV data or None if not cached or stale
        """
        cache_path = self._get_ohlcv_cache_path(symbol, timeframe)
        cache_key = f"{symbol}_{timeframe}"
        
        if not cache_path.exists():
            logger.debug(f"No cached OHLCV data found for {symbol} ({timeframe})")
            self.metadata["cache_stats"]["misses"] += 1
            self._save_metadata()
            return None
        
        # Check if cache is stale
        if max_age_days is not None:
            last_updated = self.metadata.get("last_updated", {}).get(cache_key)
            if last_updated:
                last_updated_date = datetime.fromisoformat(last_updated)
                age = datetime.now() - last_updated_date
                if age > timedelta(days=max_age_days):
                    logger.debug(f"Cached OHLCV data for {symbol} ({timeframe}) is stale ({age.days} days old)")
                    self.metadata["cache_stats"]["misses"] += 1
                    self._save_metadata()
                    return None
            else:
                # No last_updated timestamp, consider it stale
                logger.debug(f"Cached OHLCV data for {symbol} ({timeframe}) has no timestamp, considering stale")
                self.metadata["cache_stats"]["misses"] += 1
                self._save_metadata()
                return None
        
        try:
            # Load cached data
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded cached OHLCV data for {symbol} ({timeframe}): {len(df)} records")
            self.metadata["cache_stats"]["hits"] += 1
            self._save_metadata()
            return df
        except Exception as e:
            logger.error(f"Failed to load cached OHLCV data for {symbol} ({timeframe}): {e}")
            self.metadata["cache_stats"]["misses"] += 1
            self._save_metadata()
            return None
    
    def cache_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        """Cache OHLCV data to disk
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '1d', '1h')
            df: DataFrame with OHLCV data
            
        Returns:
            True if successful, False otherwise
        """
        if df.empty:
            logger.warning(f"Attempted to cache empty OHLCV dataframe for {symbol} ({timeframe})")
            return False
        
        cache_path = self._get_ohlcv_cache_path(symbol, timeframe)
        cache_key = f"{symbol}_{timeframe}"
        
        try:
            # Save dataframe to CSV
            df.to_csv(cache_path)
            
            # Update metadata
            self.metadata["last_updated"][cache_key] = datetime.now().isoformat()
            self._save_metadata()
            
            logger.info(f"Cached {len(df)} OHLCV records for {symbol} ({timeframe})")
            return True
        except Exception as e:
            logger.error(f"Failed to cache OHLCV data for {symbol} ({timeframe}): {e}")
            return False
    
    def update_ohlcv_cache(self, symbol: str, timeframe: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """Update cached OHLCV data with new data
        
        Gets existing cached data, merges with new data, and saves the combined result.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '1d', '1h')
            new_df: DataFrame with new OHLCV data
            
        Returns:
            Combined DataFrame with both cached and new data
        """
        if new_df.empty:
            logger.warning(f"Attempted to update cache with empty dataframe for {symbol} ({timeframe})")
            cached_df = self.get_cached_ohlcv(symbol, timeframe, max_age_days=None)
            return cached_df if cached_df is not None else pd.DataFrame()
        
        # Get existing cached data if available
        cached_df = self.get_cached_ohlcv(symbol, timeframe, max_age_days=None)
        
        if cached_df is None or cached_df.empty:
            # No existing data, just cache the new data
            logger.debug(f"No existing data for {symbol} ({timeframe}), caching new data")
            self.cache_ohlcv(symbol, timeframe, new_df)
            return new_df
        
        # Combine cached and new data
        combined_df = pd.concat([cached_df, new_df])
        
        # Remove duplicates, keeping the newest version of any duplicates
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        
        # Sort by date
        combined_df = combined_df.sort_index()
        
        # Cache the combined data
        self.cache_ohlcv(symbol, timeframe, combined_df)
        
        logger.info(f"Updated cache for {symbol} ({timeframe}): {len(combined_df)} records total")
        return combined_df
    
    def get_cached_indicators(self, symbol: str, timeframe: str, max_age_days: Optional[int] = 1) -> Optional[pd.DataFrame]:
        """Retrieve cached technical indicators if available and not too old
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '1d', '1h')
            max_age_days: Maximum age of cache in days before considering it stale
            
        Returns:
            DataFrame with technical indicators or None if not cached or stale
        """
        cache_path = self._get_indicators_cache_path(symbol, timeframe)
        cache_key = f"{symbol}_{timeframe}_indicators"
        
        if not cache_path.exists():
            logger.debug(f"No cached indicators found for {symbol} ({timeframe})")
            self.metadata["cache_stats"]["misses"] += 1
            self._save_metadata()
            return None
        
        # Check if cache is stale
        if max_age_days is not None:
            last_updated = self.metadata.get("last_updated", {}).get(cache_key)
            if last_updated:
                last_updated_date = datetime.fromisoformat(last_updated)
                age = datetime.now() - last_updated_date
                if age > timedelta(days=max_age_days):
                    logger.debug(f"Cached indicators for {symbol} ({timeframe}) is stale ({age.days} days old)")
                    self.metadata["cache_stats"]["misses"] += 1
                    self._save_metadata()
                    return None
            else:
                # No last_updated timestamp, consider it stale
                logger.debug(f"Cached indicators for {symbol} ({timeframe}) has no timestamp, considering stale")
                self.metadata["cache_stats"]["misses"] += 1
                self._save_metadata()
                return None
        
        try:
            # Load cached data
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded cached indicators for {symbol} ({timeframe}): {len(df)} records")
            self.metadata["cache_stats"]["hits"] += 1
            self._save_metadata()
            return df
        except Exception as e:
            logger.error(f"Failed to load cached indicators for {symbol} ({timeframe}): {e}")
            self.metadata["cache_stats"]["misses"] += 1
            self._save_metadata()
            return None
    
    def cache_indicators(self, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        """Cache technical indicators to disk
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '1d', '1h')
            df: DataFrame with technical indicators
            
        Returns:
            True if successful, False otherwise
        """
        if df.empty:
            logger.warning(f"Attempted to cache empty indicators dataframe for {symbol} ({timeframe})")
            return False
        
        cache_path = self._get_indicators_cache_path(symbol, timeframe)
        cache_key = f"{symbol}_{timeframe}_indicators"
        
        try:
            # Save dataframe to CSV
            df.to_csv(cache_path)
            
            # Update metadata
            self.metadata["last_updated"][cache_key] = datetime.now().isoformat()
            self._save_metadata()
            
            logger.info(f"Cached {len(df)} indicators records for {symbol} ({timeframe})")
            return True
        except Exception as e:
            logger.error(f"Failed to cache indicators for {symbol} ({timeframe}): {e}")
            return False
    
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """Clear cache files
        
        Args:
            symbol: Trading pair symbol to clear (None for all symbols)
            timeframe: Timeframe to clear (None for all timeframes)
        """
        if symbol is None and timeframe is None:
            # Clear all cache
            logger.warning("Clearing entire cache")
            for file in self.ohlcv_dir.glob("*.csv"):
                file.unlink()
            for file in self.indicators_dir.glob("*.csv"):
                file.unlink()
            
            # Reset metadata
            self.metadata["last_updated"] = {}
            self._save_metadata()
            return
        
        pattern = ""
        if symbol:
            safe_symbol = symbol.replace('/', '_')
            pattern = f"{safe_symbol}"
        if timeframe:
            pattern += f"_{timeframe}" if pattern else "*"
        pattern += "*" if pattern else "*"
        
        logger.info(f"Clearing cache for pattern: {pattern}")
        
        # Clear matching OHLCV files
        for file in self.ohlcv_dir.glob(pattern + ".csv"):
            file.unlink()
            logger.debug(f"Deleted cache file: {file}")
        
        # Clear matching indicators files
        for file in self.indicators_dir.glob(pattern + "_indicators.csv"):
            file.unlink()
            logger.debug(f"Deleted cache file: {file}")
        
        # Update metadata
        keys_to_remove = []
        for key in self.metadata["last_updated"]:
            if (symbol is None or symbol.replace('/', '_') in key) and \
               (timeframe is None or timeframe in key):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.metadata["last_updated"][key]
        
        self._save_metadata()
        logger.info(f"Cleared {len(keys_to_remove)} cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Dictionary with cache stats and metadata
        """
        # Count cached files
        ohlcv_files = list(self.ohlcv_dir.glob("*.csv"))
        indicator_files = list(self.indicators_dir.glob("*.csv"))
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in ohlcv_files)
        total_size += sum(f.stat().st_size for f in indicator_files)
        
        stats = {
            "ohlcv_files": len(ohlcv_files),
            "indicator_files": len(indicator_files),
            "total_files": len(ohlcv_files) + len(indicator_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_hits": self.metadata["cache_stats"]["hits"],
            "cache_misses": self.metadata["cache_stats"]["misses"],
            "hit_ratio": round(self.metadata["cache_stats"]["hits"] / 
                              (self.metadata["cache_stats"]["hits"] + self.metadata["cache_stats"]["misses"] or 1), 2),
            "symbols_cached": len(set(f.stem.split('_')[0] for f in ohlcv_files)),
            "last_updated": self.metadata.get("last_updated", {})
        }
        
        return stats