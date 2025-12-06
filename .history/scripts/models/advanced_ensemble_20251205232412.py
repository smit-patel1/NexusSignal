"""
Advanced ensemble methods with cross-validated stacking.

Includes:
- Cross-validated stacking with meta-features
- Residual-based meta-learning
- Dynamic weighting
- Model diversity metrics
"""

from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


def create_diverse_base_models() -> Dict[str, any]:
    """
    Create a diverse set of base models.

    Returns:
        Dictionary of model name to model instance
    """
    models = {
        # Linear models
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001, max_iter=5000),
        "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        # Tree-based models
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "hist_gbm": HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=7,
            learning_rate=0.05,
            random_state=42,
        ),
        # Gradient boosting
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        ),
        "catboost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False,
        ),
    }

    return models

def stack_prediction_dicts(pred_dicts: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert list of prediction dicts into a consistent 2D array.
    Keys are sorted alphabetically to guarantee identical column order.
    """
    model_names = sorted(pred_dicts[0].keys())
    matrix = np.column_stack([[d[m] for d in pred_dicts] for m in model_names])
    return matrix, model_names


def cross_validated_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, any],
    n_folds: int = 5,
    stratified: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Generate cross-validated out-of-fold predictions for stacking.

    Args:
        X: Features
        y: Targets
        models: Dictionary of model instances
        n_folds: Number of CV folds
        stratified: Whether to use stratified splits (for classification)

    Returns:
        DataFrame of OOF predictions and dictionary of trained models per fold
    """
    n_samples = len(X)
    oof_predictions = pd.DataFrame(index=X.index)

    # Store trained models for each fold
    fold_models = {name: [] for name in models.keys()}

    # K-Fold split
    kf = KFold(n_splits=n_folds, shuffle=False)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        # FIX: Ensure no NaN/inf values - convert to clean numpy arrays
        X_train_arr = np.nan_to_num(X_train.values, nan=0.0, posinf=0.0, neginf=0.0)
        y_train_arr = np.nan_to_num(y_train.values, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_arr = np.nan_to_num(X_val.values, nan=0.0, posinf=0.0, neginf=0.0)
        y_val_arr = np.nan_to_num(y.iloc[val_idx].values, nan=0.0, posinf=0.0, neginf=0.0)

        # Train each model
        for model_name, model in models.items():
            # Clone model
            from sklearn.base import clone

            model_clone = clone(model)

            # Fit
            try:
                if hasattr(model_clone, "fit"):
                    if model_name in ["lightgbm", "xgboost", "catboost"]:
                        # Early stopping for gradient boosting
                        if model_name == "lightgbm":
                            model_clone.fit(
                                X_train_arr,
                                y_train_arr,
                                eval_set=[(X_val_arr, y_val_arr)],
                                callbacks=[lgb.early_stopping(50, verbose=False)],
                            )
                        elif model_name == "xgboost":
                            model_clone.set_params(early_stopping_rounds=50)
                            model_clone.fit(
                                X_train_arr,
                                y_train_arr,
                                eval_set=[(X_val_arr, y_val_arr)],
                                verbose=False,
                            )
                        elif model_name == "catboost":
                            model_clone.fit(
                                X_train_arr,
                                y_train_arr,
                                eval_set=(X_val_arr, y_val_arr),
                                use_best_model=True,
                                early_stopping_rounds=50,
                            )
                    else:
                        model_clone.fit(X_train_arr, y_train_arr)

                    # Predict on validation set
                    val_pred = model_clone.predict(X_val_arr)

                    # Store predictions
                    if model_name not in oof_predictions.columns:
                        oof_predictions[model_name] = np.nan

                    oof_predictions.loc[X_val.index, model_name] = val_pred

                    # Store trained model
                    fold_models[model_name].append(model_clone)

            except Exception as e:
                warnings.warn(f"Model {model_name} failed on fold {fold_idx}: {e}")
                continue

    return oof_predictions, fold_models


def compute_meta_features(
    base_predictions: pd.DataFrame, y_true: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute meta-features from base model predictions.

    Args:
        base_predictions: DataFrame of base model predictions
        y_true: True targets

    Returns:
        DataFrame of meta-features
    """
    meta_features = pd.DataFrame(index=base_predictions.index)

    # Original predictions
    for col in base_predictions.columns:
        meta_features[col] = base_predictions[col]

    # Statistical aggregations (skip NaN values)
    meta_features["pred_mean"] = base_predictions.mean(axis=1, skipna=True)
    meta_features["pred_std"] = base_predictions.std(axis=1, skipna=True)
    meta_features["pred_median"] = base_predictions.median(axis=1, skipna=True)
    meta_features["pred_min"] = base_predictions.min(axis=1, skipna=True)
    meta_features["pred_max"] = base_predictions.max(axis=1, skipna=True)
    meta_features["pred_range"] = meta_features["pred_max"] - meta_features["pred_min"]

    # Residuals (if y_true provided)
    if y_true is not None:
        for col in base_predictions.columns:
            residual = y_true - base_predictions[col]
            meta_features[f"{col}_residual"] = residual
            meta_features[f"{col}_abs_residual"] = np.abs(residual)

        # Mean residual across models
        residuals = pd.DataFrame(
            {f"{col}_residual": y_true - base_predictions[col] for col in base_predictions.columns}
        )
        meta_features["mean_residual"] = residuals.mean(axis=1)
        meta_features["std_residual"] = residuals.std(axis=1)

    # Diversity measures
    # Pairwise correlations (measure of disagreement)
    if len(base_predictions.columns) > 1:
        pairwise_diffs = []
        cols = list(base_predictions.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                diff = np.abs(base_predictions[cols[i]] - base_predictions[cols[j]])
                pairwise_diffs.append(diff)

        if pairwise_diffs:
            meta_features["avg_pairwise_diff"] = pd.DataFrame(pairwise_diffs).T.mean(axis=1)
    
    meta_features = meta_features.reindex(sorted(meta_features.columns), axis=1)
    return meta_features


def train_stacking_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    base_models: Dict[str, any],
    meta_learner=None,
    n_folds: int = 5,
    use_meta_features: bool = True,
) -> Dict[str, any]:
    """
    Train a stacking ensemble with cross-validated base predictions.

    Args:
        X: Training features
        y: Training targets
        base_models: Dictionary of base model instances
        meta_learner: Meta-learner model (if None, uses CatBoost)
        n_folds: Number of CV folds
        use_meta_features: Whether to augment with meta-features

    Returns:
        Dictionary with trained ensemble artifacts
    """
    # FIX: Aggressively clean input data - handle NaN/inf values
    X_clean = X.copy()
    
    # Step 1: Replace inf with NaN
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    
    # Step 2: Drop columns that are entirely NaN or have high NaN ratio
    nan_ratio = X_clean.isna().mean()
    valid_cols = nan_ratio[nan_ratio < 0.5].index.tolist()
    X_clean = X_clean[valid_cols]
    
    # Step 3: Fill remaining NaN with 0 (safest approach)
    X_clean = X_clean.fillna(0.0)
    
    # Step 4: Drop any rows that still have NaN (shouldn't happen but be safe)
    valid_rows = ~X_clean.isna().any(axis=1)
    X_clean = X_clean.loc[valid_rows]
    
    # Step 5: Final safety check - convert to numpy and back to ensure clean
    X_values = X_clean.values
    if np.any(np.isnan(X_values)) or np.any(np.isinf(X_values)):
        # Nuclear option: replace all non-finite values with 0
        X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)
        X_clean = pd.DataFrame(X_values, index=X_clean.index, columns=X_clean.columns)
    
    # Align y with cleaned X
    y_clean_input = y.loc[X_clean.index]
    
    # Get OOF predictions
    print(f"  Generating {n_folds}-fold cross-validated predictions...")
    oof_predictions, fold_models = cross_validated_predictions(
        X_clean, y_clean_input, base_models, n_folds=n_folds
    )

    # Compute meta-features
    if use_meta_features:
        print("  Computing meta-features...")
        meta_features = compute_meta_features(oof_predictions, y_clean_input)
    else:
        meta_features = oof_predictions

    # Remove rows with NaN (from failed models)
    # FIX: Drop columns (models) that have too many NaNs first
    nan_threshold = 0.5  # Drop models with >50% NaN predictions
    nan_pct = meta_features.isna().mean()
    valid_meta_cols = nan_pct[nan_pct < nan_threshold].index
    meta_features = meta_features[valid_meta_cols]

    # Then drop rows with any remaining NaN
    valid_idx = meta_features.dropna().index
    meta_features_clean = meta_features.loc[valid_idx]
    # Ensure deterministic column order for meta-learner
    meta_features_clean = meta_features_clean.reindex(sorted(meta_features_clean.columns), axis=1)

    y_clean = y_clean_input.loc[valid_idx]

    if len(meta_features_clean) < 100:
        raise ValueError(f"Too few valid samples for stacking: {len(meta_features_clean)}")

    # Train meta-learner
    # FIX: Use Ridge with strong regularization instead of CatBoost
    if meta_learner is None:
        from sklearn.linear_model import Ridge
        meta_learner = Ridge(alpha=10.0, random_state=42)

    print(f"  Training meta-learner on {len(meta_features_clean)} samples...")
    meta_learner.fit(meta_features_clean, y_clean)
    # Save consistent ordering for inference stage
    ordered_feature_names = list(meta_features_clean.columns)

    # Train final base models on full cleaned data
    print("  Training final base models on full data...")
    final_base_models = {}
    
    # Convert to numpy arrays to ensure no NaN issues
    X_train_arr = np.nan_to_num(X_clean.values, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_arr = np.nan_to_num(y_clean_input.values, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create clean DataFrame for models that need it
    X_train_clean = pd.DataFrame(X_train_arr, index=X_clean.index, columns=X_clean.columns)
    
    for model_name, model in base_models.items():
        try:
            from sklearn.base import clone

            model_clone = clone(model)
            model_clone.fit(X_train_clean, y_train_arr)
            final_base_models[model_name] = model_clone
        except Exception as e:
            warnings.warn(f"Failed to train final {model_name}: {e}")

    ensemble = {
        "feature_order": ordered_feature_names,
        "base_models": final_base_models,
        "meta_learner": meta_learner,
        "fold_models": fold_models,
        "use_meta_features": use_meta_features,
        "n_folds": n_folds,
        "valid_feature_cols": valid_cols,  # Store valid feature columns for prediction
    }

    return ensemble


def predict_with_ensemble(
    X: pd.DataFrame,
    ensemble: Dict[str, any],
    return_components: bool = False,
) -> np.ndarray:
    """
    Make predictions using trained stacking ensemble.

    Args:
        X: Features
        ensemble: Trained ensemble artifacts
        return_components: Whether to return individual model predictions

    Returns:
        Ensemble predictions (and optionally component predictions)
    """
    base_models = ensemble["base_models"]
    meta_learner = ensemble["meta_learner"]
    use_meta_features = ensemble.get("use_meta_features", False)
    
    # FIX: Apply same cleaning as during training
    X_clean = X.copy()
    
    # Use only the valid columns from training if available
    if "valid_feature_cols" in ensemble:
        valid_cols = [c for c in ensemble["valid_feature_cols"] if c in X_clean.columns]
        X_clean = X_clean[valid_cols]
    
    # Convert to clean numpy array - same as training
    X_arr = np.nan_to_num(X_clean.values, nan=0.0, posinf=0.0, neginf=0.0)

    # Get base model predictions
    base_predictions = pd.DataFrame(index=X_clean.index)
    for model_name, model in base_models.items():
        try:
            pred = model.predict(X_arr)
            base_predictions[model_name] = pred
        except Exception as e:
            warnings.warn(f"Prediction failed for {model_name}: {e}")

    # Compute meta-features if needed
    if use_meta_features:
        meta_input = compute_meta_features(base_predictions, y_true=None)
    else:
        meta_input = base_predictions

    if "feature_order" in ensemble:
        meta_input = meta_input.reindex(ensemble["feature_order"], axis=1)

    # Meta-learner prediction
    ensemble_pred = meta_learner.predict(meta_input)

    if return_components:
        return ensemble_pred, base_predictions
    else:
        return ensemble_pred


def compute_model_diversity(predictions: pd.DataFrame) -> dict:
    """
    Compute diversity metrics for ensemble models.

    Args:
        predictions: DataFrame of model predictions (columns = models)

    Returns:
        Dictionary of diversity metrics
    """
    # Pairwise correlation
    corr_matrix = predictions.corr()
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

    # Disagreement (average pairwise difference)
    n_models = len(predictions.columns)
    pairwise_diffs = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            diff = np.abs(
                predictions.iloc[:, i] - predictions.iloc[:, j]
            ).mean()
            pairwise_diffs.append(diff)

    avg_disagreement = np.mean(pairwise_diffs) if pairwise_diffs else 0.0

    # Coefficient of variation (std / mean of predictions)
    cv = (predictions.std(axis=1) / np.abs(predictions.mean(axis=1)).clip(lower=1e-10)).mean()

    return {
        "avg_correlation": avg_corr,
        "avg_disagreement": avg_disagreement,
        "coefficient_of_variation": cv,
        "n_models": n_models,
    }


def residual_based_weighting(
    y_true: np.ndarray, predictions: pd.DataFrame
) -> np.ndarray:
    """
    Compute optimal weights based on residual analysis.

    Args:
        y_true: True targets
        predictions: DataFrame of model predictions

    Returns:
        Optimal weights (sum to 1)
    """
    n_models = len(predictions.columns)

    # Compute inverse MSE weights
    mse_scores = []
    for col in predictions.columns:
        mse = np.mean((y_true - predictions[col]) ** 2)
        mse_scores.append(mse)

    # Inverse weights (lower error = higher weight)
    inv_mse = 1.0 / (np.array(mse_scores) + 1e-10)
    weights = inv_mse / inv_mse.sum()

    return weights
