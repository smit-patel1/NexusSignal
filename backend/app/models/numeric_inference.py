"""
Inference API for trained numeric ensemble models.

Functions:
- load_numeric_model(): Load trained model artifacts
- predict_numeric_returns(): Generate return predictions
- get_model_info(): Get model metadata
- list_available_models(): List all trained models
"""

from pathlib import Path
from typing import Dict, Optional, Union
import warnings

import joblib
import pandas as pd
import numpy as np

from scripts.models.advanced_preprocessing import inverse_preprocess_targets
from scripts.models.advanced_ensemble import predict_with_ensemble

MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models" / "numeric"

_MODEL_CACHE = {}


def load_numeric_model(ticker: str, force_reload: bool = False) -> Optional[Dict]:
    """
    Load trained ensemble artifacts for a ticker.

    Args:
        ticker: Stock ticker symbol
        force_reload: Force reload from disk (ignore cache)

    Returns:
        Model artifacts dict or None if not found
    """
    if not force_reload and ticker in _MODEL_CACHE:
        return _MODEL_CACHE[ticker]

    model_path = MODELS_DIR / f"{ticker}_numeric_ensemble.pkl"

    if not model_path.exists():
        warnings.warn(f"Model not found for {ticker}: {model_path}")
        return None

    try:
        artifacts = joblib.load(model_path)
        _MODEL_CACHE[ticker] = artifacts
        return artifacts
    except Exception as e:
        warnings.warn(f"Failed to load model for {ticker}: {e}")
        return None


def predict_numeric_returns(
    ticker: str,
    feature_row: Union[pd.Series, pd.DataFrame],
    use_ensemble: bool = True,
) -> Dict[str, Union[float, Dict]]:
    """
    Predict future returns using trained ensemble.

    Args:
        ticker: Stock ticker symbol
        feature_row: Single row of features (Series or 1-row DataFrame)
        use_ensemble: Use stacking ensemble (True) or best single model (False)

    Returns:
        {
            "target_1h_return": float,
            "target_4h_return": float,
            "target_24h_return": float,
            "components": {
                "model_name": {"1h": float, "4h": float, "24h": float},
                ...
            }
        }
    """
    artifacts = load_numeric_model(ticker)

    if artifacts is None:
        raise ValueError(f"Model not found for {ticker}")

    if isinstance(feature_row, pd.Series):
        feature_row = feature_row.to_frame().T

    predictions = {}
    components = {}

    for target_name, horizon_artifacts in artifacts["horizons"].items():
        horizon_short = target_name.replace("target_", "").replace("_return", "")

        feature_columns = horizon_artifacts["feature_columns"]
        missing_cols = [c for c in feature_columns if c not in feature_row.columns]

        if missing_cols:
            warnings.warn(
                f"Missing features for {ticker} {horizon_short}: {missing_cols[:5]}"
            )
            predictions[target_name] = np.nan
            continue

        features_aligned = feature_row[feature_columns]

        scaler = horizon_artifacts["scaler"]
        features_scaled = pd.DataFrame(
            scaler.transform(features_aligned),
            columns=feature_columns,
            index=features_aligned.index,
        )

        transform_params = horizon_artifacts["transform_params"]

        if use_ensemble and "stacking_ensemble" in horizon_artifacts:
            ensemble = horizon_artifacts["stacking_ensemble"]

            try:
                pred_scaled = predict_with_ensemble(
                    features_scaled, ensemble, return_components=False
                )
                pred = inverse_preprocess_targets(pred_scaled, transform_params)
                predictions[target_name] = float(pred[0])

                _, base_preds = predict_with_ensemble(
                    features_scaled, ensemble, return_components=True
                )

                for model_name in base_preds.columns:
                    if model_name not in components:
                        components[model_name] = {}
                    pred_scaled_comp = base_preds[model_name].values
                    pred_comp = inverse_preprocess_targets(
                        pred_scaled_comp, transform_params
                    )
                    components[model_name][horizon_short] = float(pred_comp[0])

            except Exception as e:
                warnings.warn(f"Ensemble prediction failed for {horizon_short}: {e}")
                predictions[target_name] = np.nan

        else:
            best_model_name = None
            best_metric = -np.inf

            for model_name, metrics in horizon_artifacts["metrics"].items():
                if model_name == "stacking_ensemble":
                    continue
                r2 = metrics.get("r2", -np.inf)
                if r2 > best_metric:
                    best_metric = r2
                    best_model_name = model_name

            if best_model_name and best_model_name in horizon_artifacts["base_models"]:
                model = horizon_artifacts["base_models"][best_model_name]
                pred_scaled = model.predict(features_scaled)
                pred = inverse_preprocess_targets(pred_scaled, transform_params)
                predictions[target_name] = float(pred[0])

                if best_model_name not in components:
                    components[best_model_name] = {}
                components[best_model_name][horizon_short] = float(pred[0])
            else:
                predictions[target_name] = np.nan

    result = {
        "target_1h_return": predictions.get("target_1h_return", np.nan),
        "target_4h_return": predictions.get("target_4h_return", np.nan),
        "target_24h_return": predictions.get("target_24h_return", np.nan),
        "components": components,
    }

    return result


def get_model_info(ticker: str) -> Dict[str, any]:
    """
    Get metadata about a trained model.

    Args:
        ticker: Stock ticker symbol

    Returns:
        {
            "ticker": str,
            "horizons": list,
            "num_features": int,
            "has_neural": bool,
            "base_models": list,
        }
    """
    artifacts = load_numeric_model(ticker)

    if artifacts is None:
        return {}

    horizons_info = {}

    for target_name, horizon_artifacts in artifacts["horizons"].items():
        horizon_short = target_name.replace("target_", "").replace("_return", "")

        horizons_info[horizon_short] = {
            "num_features": len(horizon_artifacts["feature_columns"]),
            "has_neural": horizon_artifacts.get("has_neural", False),
            "base_models": list(horizon_artifacts["base_models"].keys()),
            "has_stacking": "stacking_ensemble" in horizon_artifacts,
        }

    return {
        "ticker": ticker,
        "horizons": list(horizons_info.keys()),
        "horizon_details": horizons_info,
    }


def list_available_models() -> list:
    """
    List all trained models.

    Returns:
        List of ticker symbols with trained models
    """
    if not MODELS_DIR.exists():
        return []

    model_files = list(MODELS_DIR.glob("*_numeric_ensemble.pkl"))
    tickers = [f.stem.replace("_numeric_ensemble", "") for f in model_files]

    return sorted(tickers)


def clear_model_cache() -> None:
    """Clear the model cache."""
    global _MODEL_CACHE
    _MODEL_CACHE = {}
