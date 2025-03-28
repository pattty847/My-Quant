# quant_system/analysis/advanced_models.py
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
import seaborn as sns
from datetime import datetime

from quant_system.utils import get_logger

# Initialize logger
logger = get_logger("analysis.advanced_models")

class DimensionalityReducer:
    """PCA analysis for technical indicators"""
    
    def __init__(self, output_dir=None):
        """Initialize the PCA analysis tool
        
        Args:
            output_dir: Optional directory to save visualizations
        """
        self.scaler = StandardScaler()
        self.pca = None
        self.output_dir = output_dir
        
        # Create output directory if specified and doesn't exist
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")
        
    def fit_transform(self, df: pd.DataFrame, feature_columns: List[str] = None, 
                      n_components: int = 2, create_plots: bool = True) -> pd.DataFrame:
        """Fit PCA model to data and transform data
        
        Args:
            df: DataFrame of OHLCV data with technical indicators
            feature_columns: List of column names to use as features
            n_components: Number of principal components to extract
            create_plots: Whether to create and save PCA plots
            
        Returns:
            DataFrame with original data plus PCA components
        """
        if df.empty:
            logger.warning("Cannot run PCA: Empty dataframe provided")
            return df
        
        # Copy the dataframe
        result_df = df.copy()
        
        # Default to using all numeric columns if no feature columns specified
        if feature_columns is None:
            # Filter to only numeric columns
            feature_columns = df.select_dtypes(include=['number']).columns.tolist()
            
            # Remove any OHLCV columns from features (they're usually targets)
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in feature_columns if col.lower() not in ohlcv_columns]
            logger.debug(f"Automatically selected {len(feature_columns)} feature columns")
        
        # Check if we have enough feature columns
        if len(feature_columns) < 2:
            logger.error(f"Need at least 2 feature columns for PCA, got {len(feature_columns)}")
            return result_df
        
        # Check for missing values
        if df[feature_columns].isna().any().any():
            logger.warning("Input data contains NaN values. Filling with forward/backward fill")
            # First forward fill, then backward fill to handle any remaining NaNs
            feature_data = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
        else:
            feature_data = df[feature_columns]
            
        # Standardize features
        logger.debug(f"Standardizing {len(feature_columns)} features")
        features_scaled = self.scaler.fit_transform(feature_data)
        
        # Apply PCA
        logger.debug(f"Running PCA with {n_components} components")
        self.pca = PCA(n_components=n_components)
        components = self.pca.fit_transform(features_scaled)
        
        # Add PCA components to the original dataframe
        for i in range(n_components):
            result_df[f'PC{i+1}'] = components[:, i]
            
        logger.info(f"PCA completed: {n_components} components explain {self.pca.explained_variance_ratio_.sum()*100:.2f}% of variance")
        
        # Create plots if requested
        if create_plots and self.output_dir:
            self._create_pca_plots(result_df, feature_columns)
            
        return result_df
    
    def transform_new(self, df: pd.DataFrame, feature_columns: List[str] = None) -> pd.DataFrame:
        """Transform new data using existing PCA model
        
        Args:
            df: DataFrame with technical indicators
            feature_columns: List of column names to use (must match training data)
            
        Returns:
            DataFrame with original data plus PCA components
        """
        if self.pca is None:
            logger.error("PCA model not fitted. Call fit_transform first")
            return df
            
        if df.empty:
            logger.warning("Cannot transform: Empty dataframe provided")
            return df
            
        # Copy the dataframe
        result_df = df.copy()
        
        # Use same feature columns as in training if not specified
        if feature_columns is None:
            # Filter to only numeric columns
            feature_columns = df.select_dtypes(include=['number']).columns.tolist()
            
            # Remove any OHLCV columns from features
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in feature_columns if col.lower() not in ohlcv_columns]
            
        # Check for missing values
        if df[feature_columns].isna().any().any():
            logger.warning("Input data contains NaN values. Filling with forward/backward fill")
            feature_data = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
        else:
            feature_data = df[feature_columns]
            
        # Standardize features using saved scaler
        features_scaled = self.scaler.transform(feature_data)
        
        # Apply PCA transformation
        components = self.pca.transform(features_scaled)
        
        # Add PCA components to the original dataframe
        for i in range(self.pca.n_components_):
            result_df[f'PC{i+1}'] = components[:, i]
            
        logger.info(f"Transformed new data with PCA model")
        return result_df
    
    def get_feature_importance(self) -> Dict[int, Dict[str, float]]:
        """Get feature importance for each principal component
        
        Returns:
            Dictionary of PC index to {feature: importance} mapping
        """
        if self.pca is None:
            logger.error("PCA model not fitted. Call fit_transform first")
            return {}
            
        feature_importance = {}
        for i, component in enumerate(self.pca.components_):
            # Get absolute values of component loadings
            abs_components = np.abs(component)
            # Normalize to sum to 1
            normalized_components = abs_components / abs_components.sum()
            feature_importance[i+1] = normalized_components
            
        return feature_importance
    
    def get_explained_variance(self) -> Dict[str, List[float]]:
        """Get explained variance information
        
        Returns:
            Dictionary with explained variance and cumulative explained variance
        """
        if self.pca is None:
            logger.error("PCA model not fitted. Call fit_transform first")
            return {"explained_variance_ratio": [], "cumulative_explained_variance": []}
            
        return {
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": np.cumsum(self.pca.explained_variance_ratio_).tolist()
        }
    
    def _create_pca_plots(self, df: pd.DataFrame, feature_columns: List[str]):
        """Create and save PCA visualization plots
        
        Args:
            df: DataFrame with PCA components
            feature_columns: List of original feature columns
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Feature importance plot
        plt.figure(figsize=(12, 8))
        feature_importance = []
        
        for i, component in enumerate(self.pca.components_):
            # Create a dictionary of feature names and loadings
            importance = dict(zip(feature_columns, component))
            # Sort by absolute value
            sorted_importance = {k: v for k, v in sorted(importance.items(), 
                                key=lambda x: abs(x[1]), reverse=True)}
            feature_importance.append(sorted_importance)
            
            # Plot only top features for clarity
            top_features = list(sorted_importance.keys())[:10]  
            values = [sorted_importance[feature] for feature in top_features]
            
            plt.subplot(1, self.pca.n_components_, i+1)
            colors = ['green' if v > 0 else 'red' for v in values]
            plt.barh(top_features, [abs(v) for v in values], color=colors)
            plt.title(f'PC{i+1} Feature Importance')
            plt.xlabel('Importance (Absolute Value)')
            
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pca_feature_importance_{timestamp}.png")
        plt.close()
        
        # 2. Explained variance plot
        plt.figure(figsize=(10, 6))
        explained_var = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.6, label='Individual')
        plt.step(range(1, len(cumulative_var) + 1), cumulative_var, where='mid', 
                 label=f'Cumulative ({cumulative_var[-1]:.2%})')
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
        plt.title('Explained Variance by Principal Components')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(range(1, len(explained_var) + 1))
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pca_explained_variance_{timestamp}.png")
        plt.close()
        
        # 3. PC1 vs PC2 scatter plot (if we have at least 2 components)
        if self.pca.n_components_ >= 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df['PC1'], df['PC2'], c=df['close'], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Price')
            plt.title('PC1 vs PC2 Colored by Price')
            plt.xlabel(f'PC1 ({explained_var[0]:.2%} Variance)')
            plt.ylabel(f'PC2 ({explained_var[1]:.2%} Variance)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/pca_scatter_{timestamp}.png")
            plt.close()
            
            # 4. PCA components over time
            plt.figure(figsize=(12, 6))
            for i in range(min(3, self.pca.n_components_)):  # Plot up to first 3 components
                plt.plot(df.index, df[f'PC{i+1}'], label=f'PC{i+1}')
            plt.title('Principal Components Over Time')
            plt.xlabel('Date')
            plt.ylabel('Component Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/pca_time_series_{timestamp}.png")
            plt.close()
            
        logger.info(f"Created PCA visualization plots in {self.output_dir}")


class RegimeDetector:
    """Detect market regimes using clustering and HMM"""
    
    def __init__(self, output_dir=None):
        """Initialize the market regime detector
        
        Args:
            output_dir: Optional directory to save visualizations
        """
        self.kmeans = None
        self.hmm = None
        self.n_regimes = None
        self.output_dir = output_dir
        
        # Create output directory if specified and doesn't exist
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")
            
    def detect_regimes_kmeans(self, df: pd.DataFrame, feature_columns: List[str], 
                             n_regimes: int = 3, create_plots: bool = True) -> pd.DataFrame:
        """Detect market regimes using KMeans clustering
        
        Args:
            df: DataFrame with technical indicators or PCA components
            feature_columns: List of column names to use for clustering
            n_regimes: Number of regimes/clusters to identify
            create_plots: Whether to create and save regime plots
            
        Returns:
            DataFrame with regime labels added
        """
        if df.empty:
            logger.warning("Cannot detect regimes: Empty dataframe provided")
            return df
            
        # Copy the dataframe
        result_df = df.copy()
        self.n_regimes = n_regimes
        
        # Check for missing values
        if df[feature_columns].isna().any().any():
            logger.warning("Input data contains NaN values. Filling with forward/backward fill")
            feature_data = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
        else:
            feature_data = df[feature_columns]
            
        # Fit KMeans model
        logger.debug(f"Fitting KMeans with {n_regimes} clusters on {', '.join(feature_columns)}")
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init='auto')
        regime_labels = self.kmeans.fit_predict(feature_data)
        
        # Add regime labels to the dataframe
        result_df['kmeans_regime'] = regime_labels
        
        logger.info(f"KMeans clustering identified {n_regimes} market regimes")
        
        # Create plots if requested
        if create_plots and self.output_dir:
            self._create_kmeans_plots(result_df, feature_columns)
            
        return result_df
    
    def detect_regimes_hmm(self, df: pd.DataFrame, feature_columns: List[str], 
                          n_regimes: int = 3, n_iter: int = 1000, 
                          create_plots: bool = True) -> pd.DataFrame:
        """Detect market regimes using Hidden Markov Model
        
        Args:
            df: DataFrame with technical indicators or PCA components
            feature_columns: List of column names to use for HMM
            n_regimes: Number of HMM states
            n_iter: Number of iterations for HMM fitting
            create_plots: Whether to create and save regime plots
            
        Returns:
            DataFrame with HMM state labels added
        """
        if df.empty:
            logger.warning("Cannot detect regimes with HMM: Empty dataframe provided")
            return df
            
        # Copy the dataframe
        result_df = df.copy()
        self.n_regimes = n_regimes
        
        # Check for missing values
        if df[feature_columns].isna().any().any():
            logger.warning("Input data contains NaN values. Filling with forward/backward fill")
            feature_data = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
        else:
            feature_data = df[feature_columns]
            
        # Fit HMM model
        logger.debug(f"Fitting HMM with {n_regimes} states on {', '.join(feature_columns)}")
        self.hmm = GaussianHMM(n_components=n_regimes, covariance_type='full', 
                              n_iter=n_iter, random_state=42)
        
        try:
            self.hmm.fit(feature_data)
            state_labels = self.hmm.predict(feature_data)
            
            # Add HMM state labels to the dataframe
            result_df['hmm_regime'] = state_labels
            
            logger.info(f"HMM identified {n_regimes} market regimes after {self.hmm.monitor_.iter} iterations")
            
            # Create plots if requested
            if create_plots and self.output_dir:
                self._create_hmm_plots(result_df, feature_columns)
                
        except Exception as e:
            logger.error(f"Error fitting HMM: {str(e)}")
            
        return result_df
    
    def predict_regime_kmeans(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Predict regimes for new data using the fitted KMeans model
        
        Args:
            df: DataFrame with technical indicators or PCA components
            feature_columns: List of column names (must match training data)
            
        Returns:
            DataFrame with regime predictions added
        """
        if self.kmeans is None:
            logger.error("KMeans model not fitted. Call detect_regimes_kmeans first")
            return df
            
        if df.empty:
            logger.warning("Cannot predict regimes: Empty dataframe provided")
            return df
            
        # Copy the dataframe
        result_df = df.copy()
        
        # Check for missing values
        if df[feature_columns].isna().any().any():
            logger.warning("Input data contains NaN values. Filling with forward/backward fill")
            feature_data = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
        else:
            feature_data = df[feature_columns]
            
        # Predict regimes
        regime_labels = self.kmeans.predict(feature_data)
        
        # Add regime labels to the dataframe
        result_df['kmeans_regime'] = regime_labels
        
        logger.info(f"KMeans predicted regimes for {len(df)} data points")
        return result_df
    
    def predict_regime_hmm(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Predict regimes for new data using the fitted HMM model
        
        Args:
            df: DataFrame with technical indicators or PCA components
            feature_columns: List of column names (must match training data)
            
        Returns:
            DataFrame with regime predictions added
        """
        if self.hmm is None:
            logger.error("HMM model not fitted. Call detect_regimes_hmm first")
            return df
            
        if df.empty:
            logger.warning("Cannot predict HMM regimes: Empty dataframe provided")
            return df
            
        # Copy the dataframe
        result_df = df.copy()
        
        # Check for missing values
        if df[feature_columns].isna().any().any():
            logger.warning("Input data contains NaN values. Filling with forward/backward fill")
            feature_data = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
        else:
            feature_data = df[feature_columns]
            
        try:
            # Predict states
            state_labels = self.hmm.predict(feature_data)
            
            # Add state labels to the dataframe
            result_df['hmm_regime'] = state_labels
            
            logger.info(f"HMM predicted regimes for {len(df)} data points")
        except Exception as e:
            logger.error(f"Error predicting HMM states: {str(e)}")
            
        return result_df
    
    def analyze_regimes(self, df: pd.DataFrame, regime_column: str) -> Dict[int, Dict[str, Any]]:
        """Calculate statistics for each detected regime
        
        Args:
            df: DataFrame with regime labels
            regime_column: Column name containing regime labels
            
        Returns:
            Dictionary with regime statistics
        """
        if regime_column not in df.columns:
            logger.error(f"Regime column '{regime_column}' not found in dataframe")
            return {}
            
        regime_stats = {}
        
        # Calculate stats for each regime
        for regime in sorted(df[regime_column].unique()):
            regime_df = df[df[regime_column] == regime]
            
            # Skip if we have too few data points
            if len(regime_df) < 5:
                logger.warning(f"Regime {regime} has fewer than 5 data points, skipping stats")
                continue
                
            # Basic stats
            stats = {
                'count': len(regime_df),
                'avg_close': regime_df['close'].mean(),
                'avg_return': regime_df['close'].pct_change().mean() * 100,
                'volatility': regime_df['close'].pct_change().std() * 100,
                'win_rate': (regime_df['close'].pct_change() > 0).mean() * 100,
                'longest_duration': 0,
                'regime_durations': [],
                'regime_changes': []
            }
            
            # Calculate durations and transitions
            current_duration = 1
            current_start = None
            
            for i in range(1, len(df)):
                prev_regime = df[regime_column].iloc[i-1]
                curr_regime = df[regime_column].iloc[i]
                
                if prev_regime == regime and curr_regime != regime:
                    # Regime ended
                    if current_duration > 1:
                        stats['regime_durations'].append(current_duration)
                        if current_duration > stats['longest_duration']:
                            stats['longest_duration'] = current_duration
                    
                    # Record regime change
                    stats['regime_changes'].append({
                        'from': regime,
                        'to': curr_regime,
                        'date': df.index[i].strftime('%Y-%m-%d') if hasattr(df.index[i], 'strftime') else str(df.index[i])
                    })
                    
                    current_duration = 0
                
                if prev_regime != regime and curr_regime == regime:
                    # Regime started
                    current_duration = 1
                    current_start = i
                
                if prev_regime == regime and curr_regime == regime:
                    # Continuing in regime
                    current_duration += 1
            
            # Calculate average duration if we have any duration data
            if stats['regime_durations']:
                stats['avg_duration'] = sum(stats['regime_durations']) / len(stats['regime_durations'])
            else:
                stats['avg_duration'] = 0
                
            # For the last segment if we're still in this regime
            if df[regime_column].iloc[-1] == regime and current_duration > 0:
                stats['regime_durations'].append(current_duration)
                if current_duration > stats['longest_duration']:
                    stats['longest_duration'] = current_duration
            
            # Calculate technical indicator averages if available
            indicator_columns = [col for col in df.columns if col not in 
                                ['open', 'high', 'low', 'close', 'volume', regime_column]]
            
            stats['indicators'] = {}
            for col in indicator_columns:
                if col in df.columns:
                    stats['indicators'][col] = regime_df[col].mean()
            
            regime_stats[int(regime)] = stats
            
        logger.info(f"Analyzed {len(regime_stats)} regimes")
        return regime_stats
    
    def _create_kmeans_plots(self, df: pd.DataFrame, feature_columns: List[str]):
        """Create and save KMeans regime visualization plots
        
        Args:
            df: DataFrame with KMeans regime labels
            feature_columns: List of feature columns used
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Regime scatter plot (if we have PC1 and PC2 columns)
        if 'PC1' in df.columns and 'PC2' in df.columns:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df['PC1'], df['PC2'], c=df['kmeans_regime'], 
                                 cmap='viridis', alpha=0.6, s=30)
            plt.colorbar(scatter, label='Regime')
            plt.title('Market Regimes (KMeans Clustering)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/kmeans_regimes_scatter_{timestamp}.png")
            plt.close()
        
        # 2. Regime time series
        plt.figure(figsize=(12, 6))
        plt.scatter(df.index, df['close'], c=df['kmeans_regime'], cmap='viridis', alpha=0.7, s=30)
        plt.colorbar(label='Regime')
        plt.title('Market Regimes Over Time (KMeans)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/kmeans_regimes_time_{timestamp}.png")
        plt.close()
        
        # 3. Regime distribution
        plt.figure(figsize=(10, 6))
        regime_counts = df['kmeans_regime'].value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 1, len(regime_counts)))
        plt.bar(regime_counts.index, regime_counts.values, color=colors)
        plt.title('Distribution of Market Regimes (KMeans)')
        plt.xlabel('Regime')
        plt.ylabel('Number of Days')
        plt.xticks(regime_counts.index)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/kmeans_regime_distribution_{timestamp}.png")
        plt.close()
        
        logger.info(f"Created KMeans regime plots in {self.output_dir}")
    
    def _create_hmm_plots(self, df: pd.DataFrame, feature_columns: List[str]):
        """Create and save HMM regime visualization plots
        
        Args:
            df: DataFrame with HMM regime labels
            feature_columns: List of feature columns used
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. HMM State scatter plot (if we have PC1 and PC2 columns)
        if 'PC1' in df.columns and 'PC2' in df.columns:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df['PC1'], df['PC2'], c=df['hmm_regime'], 
                                 cmap='viridis', alpha=0.6, s=30)
            plt.colorbar(scatter, label='HMM State')
            plt.title('Market Regimes (Hidden Markov Model)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/hmm_regimes_scatter_{timestamp}.png")
            plt.close()
        
        # 2. State time series
        plt.figure(figsize=(12, 6))
        plt.scatter(df.index, df['close'], c=df['hmm_regime'], cmap='viridis', alpha=0.7, s=30)
        plt.colorbar(label='HMM State')
        plt.title('Market Regimes Over Time (HMM)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hmm_regimes_time_{timestamp}.png")
        plt.close()
        
        # 3. State distribution
        plt.figure(figsize=(10, 6))
        state_counts = df['hmm_regime'].value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 1, len(state_counts)))
        plt.bar(state_counts.index, state_counts.values, color=colors)
        plt.title('Distribution of Market Regimes (HMM)')
        plt.xlabel('HMM State')
        plt.ylabel('Number of Days')
        plt.xticks(state_counts.index)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hmm_regime_distribution_{timestamp}.png")
        plt.close()
        
        # 4. Transition matrix heatmap (if we have state sequence)
        if self.hmm is not None:
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.hmm.transmat_, annot=True, cmap='viridis', fmt='.2f',
                       xticklabels=range(self.n_regimes), yticklabels=range(self.n_regimes))
            plt.title('HMM State Transition Probabilities')
            plt.xlabel('To State')
            plt.ylabel('From State')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/hmm_transition_matrix_{timestamp}.png")
            plt.close()
        
        logger.info(f"Created HMM regime plots in {self.output_dir}")


def run_models(df, output_dir=None, n_components=2, n_regimes=3, 
              feature_columns=None, create_plots=True):
    """Run all advanced statistical models on the given data
    
    Args:
        df: DataFrame with market data and indicators
        output_dir: Directory to save visualizations
        n_components: Number of PCA components to extract
        n_regimes: Number of market regimes to identify
        feature_columns: List of indicator columns to use (None for automatic selection)
        create_plots: Whether to create and save visualization plots
        
    Returns:
        Enhanced DataFrame with models' output and dictionaries with analysis results
    """
    logger.info(f"Running advanced statistical analysis on {len(df)} data points")
    
    # Ensure output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    # Perform PCA dimensionality reduction
    reducer = DimensionalityReducer(output_dir=output_dir)
    df_pca = reducer.fit_transform(df, feature_columns=feature_columns, 
                                 n_components=n_components, create_plots=create_plots)
    
    # Get PCA summary statistics
    pca_stats = {
        'variance': reducer.get_explained_variance(),
        'feature_importance': reducer.get_feature_importance()
    }
    
    # Use PC columns for regime detection
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    
    # Detect market regimes with KMeans
    regime_detector = RegimeDetector(output_dir=output_dir)
    df_kmeans = regime_detector.detect_regimes_kmeans(
        df_pca, feature_columns=pc_columns, n_regimes=n_regimes, create_plots=create_plots
    )
    
    # Detect market regimes with HMM
    df_hmm = regime_detector.detect_regimes_hmm(
        df_kmeans, feature_columns=pc_columns, n_regimes=n_regimes, create_plots=create_plots
    )
    
    # Analyze regime statistics
    kmeans_stats = regime_detector.analyze_regimes(df_hmm, 'kmeans_regime')
    hmm_stats = regime_detector.analyze_regimes(df_hmm, 'hmm_regime')
    
    # Combine all stats
    all_stats = {
        'pca': pca_stats,
        'kmeans_regimes': kmeans_stats,
        'hmm_regimes': hmm_stats
    }
    
    logger.info("Advanced statistical analysis completed successfully")
    return df_hmm, all_stats 