"""
Unified Feature Builder for NexusSignal 2.0

Combines all feature engineering modules:
- Fractional differencing
- Microstructure features
- Entropy features
- Regime features
- Technical indicators

Outputs a clean, ML-ready feature matrix with proper temporal alignment.
"""

from typing import Optional, Dict, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd

from nexus2.data.fractional_diff import apply_fractional_diff, find_optimal_d
from nexus2.data.sampling import get_daily_vol
from nexus2.features.microstructure import build_microstructure_features
from nexus2.features.entropy import build_entropy_features
from nexus2.features.regime import build_regime_features, HMMRegimeDetector


class FeatureBuilder:
    """
    Comprehensive feature builder for financial ML.
    
    Pipeline:
    1. Load raw OHLCV data
    2. Compute returns and volatility
    3. Apply fractional differencing (for stationarity)
    4. Build technical features
    5. Build microstructure features
    6. Build entropy features
    7. Build regime features
    8. Combine and clean
    
    Example:
        >>> builder = FeatureBuilder(config)
        >>> features, d_values = builder.build_features(df)
        >>> X_train, X_test = features.iloc[:train_idx], features.iloc[train_idx:]
    """
    
    def __init__(
        self,
        apply_frac_diff: bool = True,
        frac_diff_threshold: float = 0.05,
        microstructure_windows: List[int] = [10, 20, 50],
        entropy_windows: List[int] = [20, 50, 100],
        n_regimes: int = 3,
        technical_windows: List[int] = [5, 10, 20, 50],
        vol_method: str = 'yang_zhang',
        vol_lookback: int = 20
    ):
        """
        Initialize feature builder.
        
        Args:
            apply_frac_diff: Whether to apply fractional differencing
            frac_diff_threshold: ADF p-value threshold for stationarity
            microstructure_windows: Windows for microstructure features
            entropy_windows: Windows for entropy features
            n_regimes: Number of HMM regimes
            technical_windows: Windows for technical indicators
            vol_method: Volatility estimation method
            vol_lookback: Volatility lookback period
        """
        self.apply_frac_diff = apply_frac_diff
        self.frac_diff_threshold = frac_diff_threshold
        self.microstructure_windows = microstructure_windows
        self.entropy_windows = entropy_windows
        self.n_regimes = n_regimes
        self.technical_windows = technical_windows
        self.vol_method = vol_method
        self.vol_lookback = vol_lookback
        
        # Store fitted parameters
        self.d_values_: Optional[Dict[str, float]] = None
        self.hmm_detector_: Optional[HMMRegimeDetector] = None
        
    def build_features(
        self,
        df: pd.DataFrame,
        fit_params: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Build complete feature set from OHLCV data.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            fit_params: Whether to fit parameters (False for inference)
        
        Returns:
            Tuple of (features_df, metadata_dict)
        """
        # Validate input
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        features = pd.DataFrame(index=df.index)
        metadata = {}
        
        # Step 1: Compute returns
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Step 2: Compute volatility
        volatility = get_daily_vol(
            df['close'],
            span=self.vol_lookback,
            method=self.vol_method,
            high=df['high'],
            low=df['low'],
            open_=df['open']
        )
        features['volatility'] = volatility
        
        # Step 3: Technical features
        tech_features = self._build_technical_features(df)
        features = pd.concat([features, tech_features], axis=1)
        
        # Step 4: Fractional differencing (for stationarity)
        if self.apply_frac_diff:
            price_cols = ['close', 'high', 'low', 'open']
            
            if fit_params:
                ffd_features, d_values = apply_fractional_diff(
                    df[price_cols],
                    auto_find_d=True,
                    config={
                        'd_min': 0.0,
                        'd_max': 1.0,
                        'd_step': 0.05,
                        'adf_threshold': self.frac_diff_threshold,
                        'weight_threshold': 1e-5,
                        'max_lag': 100
                    }
                )
                self.d_values_ = d_values
            else:
                if self.d_values_ is None:
                    raise ValueError("Must fit before transform (fit_params=True)")
                ffd_features, _ = apply_fractional_diff(
                    df[price_cols],
                    d_values=self.d_values_,
                    auto_find_d=False
                )
            
            features = pd.concat([features, ffd_features], axis=1)
            metadata['d_values'] = self.d_values_
        
        # Step 5: Microstructure features
        micro_features = build_microstructure_features(
            df, windows=self.microstructure_windows
        )
        features = pd.concat([features, micro_features], axis=1)
        
        # Step 6: Entropy features
        entropy_features = build_entropy_features(
            features[['returns', 'volatility']].rename(columns={'returns': 'close'}),
            price_col='close',
            windows=self.entropy_windows
        )
        features = pd.concat([features, entropy_features], axis=1)
        
        # Step 7: Regime features
        if 'returns' in features.columns and len(features.dropna()) > 100:
            if fit_params:
                self.hmm_detector_ = HMMRegimeDetector(n_regimes=self.n_regimes)
                returns_clean = features['returns'].dropna()
                try:
                    self.hmm_detector_.fit(returns_clean)
                    regime_features = self.hmm_detector_.predict_proba(returns_clean)
                    regime_features['current_regime'] = self.hmm_detector_.predict(returns_clean)
                    features = pd.concat([features, regime_features], axis=1)
                except Exception as e:
                    print(f"[WARN] HMM fitting failed: {e}")
            elif self.hmm_detector_ is not None:
                try:
                    returns_clean = features['returns'].dropna()
                    regime_features = self.hmm_detector_.predict_proba(returns_clean)
                    regime_features['current_regime'] = self.hmm_detector_.predict(returns_clean)
                    features = pd.concat([features, regime_features], axis=1)
                except Exception as e:
                    print(f"[WARN] HMM prediction failed: {e}")
        
        # Step 8: Lagged features
        lagged = self._build_lagged_features(features)
        features = pd.concat([features, lagged], axis=1)
        
        # Step 9: Clean
        features = self._clean_features(features)
        
        metadata['n_features'] = len(features.columns)
        metadata['n_samples'] = len(features)
        metadata['feature_names'] = list(features.columns)
        
        return features, metadata
    
    def _build_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build technical indicators."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        for w in self.technical_windows:
            # Moving averages
            features[f'sma_{w}'] = close.rolling(w).mean()
            features[f'ema_{w}'] = close.ewm(span=w).mean()
            
            # Price relative to MA
            features[f'close_to_sma_{w}'] = close / features[f'sma_{w}'] - 1
            
            # Momentum
            features[f'momentum_{w}'] = close / close.shift(w) - 1
            
            # Volatility
            features[f'vol_{w}'] = close.pct_change().rolling(w).std()
            
            # Volume features
            features[f'volume_ma_{w}'] = volume.rolling(w).mean()
            features[f'volume_ratio_{w}'] = volume / features[f'volume_ma_{w}']
            
            # Range
            features[f'range_{w}'] = (high - low).rolling(w).mean() / close
            
            # Rolling stats
            features[f'returns_skew_{w}'] = close.pct_change().rolling(w).skew()
            features[f'returns_kurt_{w}'] = close.pct_change().rolling(w).kurt()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['bb_upper'] = sma_20 + 2 * std_20
        features['bb_lower'] = sma_20 - 2 * std_20
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_pct'] = features['atr_14'] / close
        
        return features
    
    def _build_lagged_features(
        self,
        df: pd.DataFrame,
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """Build lagged versions of key features."""
        features = pd.DataFrame(index=df.index)
        
        key_features = ['returns', 'volatility', 'volume_ratio_20', 'rsi_14']
        available = [f for f in key_features if f in df.columns]
        
        for col in available:
            for lag in lags:
                features[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        return features
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean feature matrix."""
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with >50% NaN
        nan_pct = df.isna().mean()
        valid_cols = nan_pct[nan_pct < 0.5].index
        df = df[valid_cols]
        
        # Forward fill (limited to 5 periods)
        df = df.ffill(limit=5)
        
        # Drop remaining NaN rows
        df = df.dropna()
        
        return df
    
    def get_feature_importance(
        self,
        model,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Extract feature importance from trained model.
        
        Args:
            model: Trained model with feature_importances_ or coef_
            feature_names: List of feature names
        
        Returns:
            DataFrame with feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)


def load_and_build_features(
    data_path: Path,
    ticker: str,
    builder: Optional[FeatureBuilder] = None
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Load data and build features for a ticker.
    
    Args:
        data_path: Path to data directory
        ticker: Stock ticker
        builder: Optional pre-configured FeatureBuilder
    
    Returns:
        Tuple of (features_df, metadata)
    """
    # Load raw data
    raw_path = data_path / "raw" / "prices" / f"{ticker}_1h.parquet"
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Data not found: {raw_path}")
    
    df = pd.read_parquet(raw_path)
    
    # Ensure datetime index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    df = df.sort_index()
    
    # Standardize column names
    column_map = {
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    }
    df = df.rename(columns=column_map)
    
    # Build features
    if builder is None:
        builder = FeatureBuilder()
    
    features, metadata = builder.build_features(df)
    metadata['ticker'] = ticker
    
    return features, metadata

