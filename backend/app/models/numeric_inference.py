"""
Numeric model inference for NexusSignal backend.

Provides simple interface to load trained ensemble models and generate predictions.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import torch


# Model artifact directory
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models" / "numeric"

# Cache for loaded models
_MODEL_CACHE: Dict[str, Dict] = {}


def load_numeric_model(ticker: str, force_reload: bool = False) -> Optional[Dict]:
    """
    Load trained numeric ensemble model for a ticker.

    Args:
        ticker: Stock ticker symbol
        force_reload: If True, reload from disk even if cached

    Returns:
        Dictionary containing all model artifacts, or None if not found

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    # Check cache
    if ticker in _MODEL_CACHE and not force_reload:
        return _MODEL_CACHE[ticker]

    # Load from disk
    model_path = MODELS_DIR / f"{ticker}_numeric_ensemble.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found for {ticker} at {model_path}")

    artifact = joblib.load(model_path)

    # Cache it
    _MODEL_CACHE[ticker] = artifact

    return artifact


def predict_numeric_returns(
    ticker: str,
    feature_row: Union[pd.Series, pd.DataFrame],
    use_ensemble: bool = True
) -> Dict[str, Union[float, Dict]]:
    """
    Predict future returns for a ticker using trained ensemble.

    Args:
        ticker: Stock ticker symbol
        feature_row: Single row of features (pd.Series or 1-row DataFrame)
        use_ensemble: If True, use ensemble blender; if False, average base models

    Returns:
        Dictionary with predictions:
        {
            "target_1h_return": float,
            "target_4h_return": float,
            "target_24h_return": float,
            "components": {
                "lightgbm": {"1h": float, "4h": float, "24h": float},
                "xgboost": {...},
                "catboost": {...},
                "lstm": {...} or None,
                "tcn": {...} or None,
                "ensemble": {...}
            }
        }

    Raises:
        FileNotFoundError: If model doesn't exist
        ValueError: If features don't match training
    """
    # Load model
    artifact = load_numeric_model(ticker)

    # Convert to DataFrame if Series
    if isinstance(feature_row, pd.Series):
        feature_row = feature_row.to_frame().T

    # Validate features
    expected_features = artifact['feature_columns']

    # Check for missing features
    missing_features = set(expected_features) - set(feature_row.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Reorder to match training
    feature_row = feature_row[expected_features]

    # Scale features
    scaler = artifact['scaler']
    feature_scaled = scaler.transform(feature_row)
    feature_scaled_df = pd.DataFrame(
        feature_scaled,
        columns=expected_features
    )

    # Prepare output
    predictions = {}
    components = {}

    # For each horizon
    for target, horizon_artifacts in artifact['horizons'].items():
        horizon_short = target.replace('target_', '').replace('_return', '')

        # Get base model predictions
        base_preds = {}

        # Gradient boosting models
        for model_name in ['lightgbm', 'xgboost', 'catboost']:
            if model_name in horizon_artifacts['base_models']:
                model = horizon_artifacts['base_models'][model_name]
                pred = model.predict(feature_scaled_df)[0]
                base_preds[model_name] = pred

                if model_name not in components:
                    components[model_name] = {}
                components[model_name][horizon_short] = float(pred)

        # Neural models (if available)
        # Note: For neural models, we'd need the full lookback window sequence
        # For now, we'll skip them in inference and rely on gradient boosting models
        # This can be extended in future phases when we have streaming context

        if horizon_artifacts.get('has_neural', False):
            # Placeholder for LSTM/TCN
            # In production, you'd need to maintain a rolling window buffer
            components['lstm'] = None
            components['tcn'] = None

        # Ensemble prediction
        if use_ensemble and 'blender' in horizon_artifacts:
            blender = horizon_artifacts['blender']

            # Stack base predictions
            stacked_preds = np.array([base_preds[k] for k in sorted(base_preds.keys())]).reshape(1, -1)

            ensemble_pred = blender.predict(stacked_preds)[0]
        else:
            # Simple average as fallback
            ensemble_pred = np.mean(list(base_preds.values()))

        predictions[target] = float(ensemble_pred)

        if 'ensemble' not in components:
            components['ensemble'] = {}
        components['ensemble'][horizon_short] = float(ensemble_pred)

    return {
        'target_1h_return': predictions.get('target_1h_return'),
        'target_4h_return': predictions.get('target_4h_return'),
        'target_24h_return': predictions.get('target_24h_return'),
        'components': components
    }


def get_model_info(ticker: str) -> Dict[str, any]:
    """
    Get metadata about a trained model.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with model information:
        {
            'ticker': str,
            'feature_count': int,
            'feature_columns': list,
            'has_neural': bool,
            'horizons': list,
            'base_models': list
        }

    Raises:
        FileNotFoundError: If model doesn't exist
    """
    artifact = load_numeric_model(ticker)

    # Collect info
    horizons = list(artifact['horizons'].keys())
    base_models = set()

    for horizon_artifacts in artifact['horizons'].values():
        base_models.update(horizon_artifacts['base_models'].keys())

    has_neural = any(
        horizon_artifacts.get('has_neural', False)
        for horizon_artifacts in artifact['horizons'].values()
    )

    return {
        'ticker': ticker,
        'feature_count': len(artifact['feature_columns']),
        'feature_columns': artifact['feature_columns'],
        'has_neural': has_neural,
        'horizons': horizons,
        'base_models': sorted(list(base_models)),
        'lookback_window': artifact['horizons'][horizons[0]].get('lookback_window') if has_neural else None
    }


def clear_model_cache(ticker: Optional[str] = None) -> None:
    """
    Clear cached models.

    Args:
        ticker: If provided, clear only this ticker; otherwise clear all
    """
    global _MODEL_CACHE

    if ticker is None:
        _MODEL_CACHE.clear()
    elif ticker in _MODEL_CACHE:
        del _MODEL_CACHE[ticker]


def list_available_models() -> list:
    """
    List all available trained models.

    Returns:
        List of ticker symbols with trained models
    """
    if not MODELS_DIR.exists():
        return []

    model_files = MODELS_DIR.glob("*_numeric_ensemble.pkl")
    tickers = [f.stem.replace('_numeric_ensemble', '') for f in model_files]

    return sorted(tickers)
