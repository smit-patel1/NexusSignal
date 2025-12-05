"""
Phase 3: Advanced Numeric Model Training Pipeline

Trains per-ticker ensemble models to predict:
- target_1h_return
- target_4h_return
- target_24h_return

Architecture:
- Linear: Ridge, Lasso, ElasticNet
- Trees: RandomForest, HistGradientBoosting
- Boosting: LightGBM, XGBoost, CatBoost
- Neural (optional): DeepResidualLSTM, AdvancedTCN
- Ensemble: Cross-validated stacking with meta-features
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from sklearn.preprocessing import RobustScaler

sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.app.config import TICKERS
from scripts.utils.parquet_utils import read_parquet
from scripts.models.advanced_preprocessing import (
    filter_bad_data,
    preprocess_targets,
    inverse_preprocess_targets,
    robust_scale_features,
)
from scripts.models.advanced_features import build_alpha_features
from scripts.models.advanced_neural_models import (
    DeepResidualLSTM,
    AdvancedTCN,
    TimeSeriesDataset,
)
from scripts.models.advanced_ensemble import (
    create_diverse_base_models,
    train_stacking_ensemble,
    predict_with_ensemble,
)
from scripts.models.utils import create_sequence_dataset, compute_metrics, print_metrics

# ======================================================================
# CONFIGURATION
# ======================================================================

TRAIN_NEURAL = True
NEURAL_TICKERS = TICKERS[:10]
LOOKBACK_WINDOW = 32

USE_KALMAN_FILTER = False
TARGET_TRANSFORM = "log"
NORMALIZE_TARGETS = True

FILTER_BAD_DATA = True
MIN_VOLUME_PERCENTILE = 1.0
MAX_PRICE_JUMP = 20.0

BUILD_ALPHA_FEATURES = True
INCLUDE_INTERACTIONS = True

USE_STACKING = True
USE_META_FEATURES = True
STACKING_FOLDS = 5

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FEATURES_DIR = DATA_DIR / "processed" / "features"
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "numeric"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["target_1h_return", "target_4h_return", "target_24h_return"]

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================


def create_horizon_specific_features(
    df: pd.DataFrame, horizon: str
) -> pd.DataFrame:
    """Create horizon-specific features without dropping rows."""
    df_horizon = df.copy()

    if horizon == "1h":
        periods = [1, 2, 3]
    elif horizon == "4h":
        periods = [4, 8, 12]
    elif horizon == "24h":
        periods = [24, 48, 72]
    else:
        periods = [1]

    if "close" in df_horizon.columns:
        for period in periods:
            df_horizon[f"return_{period}p"] = df_horizon["close"].pct_change(period)
            df_horizon[f"volatility_{period}p"] = (
                df_horizon["close"].pct_change().rolling(period).std()
            )

        base_period = periods[0]
        df_horizon[f"momentum_{horizon}"] = (
            df_horizon["close"] / df_horizon["close"].shift(base_period) - 1
        )

        delta = df_horizon["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(base_period * 2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(base_period * 2).mean()
        rs = gain / loss.replace(0, 1e-10)
        df_horizon[f"rsi_{horizon}"] = 100 - (100 / (1 + rs))

    return df_horizon


def train_gradient_boosting_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, any]:
    """Train all base models with early stopping."""
    base_models = create_diverse_base_models()
    trained_models = {}
    val_preds = {}

    for model_name, model in base_models.items():
        print(f"    Training {model_name}...")

        try:
            if model_name == "lightgbm":
                import lightgbm as lgb
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            elif model_name == "xgboost":
                model.set_params(early_stopping_rounds=50)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            elif model_name == "catboost":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=50,
                )
            else:
                model.fit(X_train, y_train)

            trained_models[model_name] = model
            val_preds[model_name] = model.predict(X_val)

        except Exception as e:
            print(f"      [WARN] {model_name} failed: {e}")
            continue

    return {"models": trained_models, "val_preds": val_preds}


def train_neural_model(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    n_features: int,
    model_type: str = "deep_lstm",
) -> Tuple[Optional[pl.LightningModule], Optional[np.ndarray]]:
    """Train advanced neural model."""
    train_dataset = TimeSeriesDataset(
        torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)
    )
    val_dataset = TimeSeriesDataset(
        torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    if model_type == "deep_lstm":
        print("    Training DeepResidualLSTM...")
        model = DeepResidualLSTM(
            n_features=n_features,
            hidden_sizes=[128, 128, 64],
            dropout=0.3,
            learning_rate=0.001,
            use_attention=True,
        )
    elif model_type == "advanced_tcn":
        print("    Training AdvancedTCN...")
        model = AdvancedTCN(
            n_features=n_features,
            num_channels=[64, 128, 128, 64],
            kernel_size=3,
            dropout=0.3,
            learning_rate=0.001,
            use_attention=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=False,
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[early_stop],
    )

    trainer.fit(model, train_loader, val_loader)

    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            preds = model(batch_X)
            val_preds_list.append(preds.cpu().numpy())

    val_preds = np.concatenate(val_preds_list)

    return model, val_preds


def train_ticker_models(
    ticker: str,
    train_neural: bool = TRAIN_NEURAL,
) -> Optional[Dict]:
    """Training pipeline for a single ticker."""
    print("\n" + "=" * 80)
    print(f"Training models for {ticker}")
    print("=" * 80)

    feature_path = FEATURES_DIR / f"{ticker}_1h.parquet"
    if not feature_path.exists():
        print(f"[SKIP] No feature file found for {ticker}")
        return None

    df = read_parquet(feature_path)

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    df = df.sort_index()

    missing_targets = [t for t in TARGETS if t not in df.columns]
    if missing_targets:
        print(f"[SKIP] Missing targets: {missing_targets}")
        return None

    if FILTER_BAD_DATA:
        original_len = len(df)
        df = filter_bad_data(
            df,
            price_col="close",
            volume_col="volume" if "volume" in df.columns else None,
            min_volume_percentile=MIN_VOLUME_PERCENTILE,
            max_price_jump_pct=MAX_PRICE_JUMP,
        )
        filtered_count = original_len - len(df)
        if filtered_count > 0:
            print(f"  Filtered {filtered_count} bad data points ({filtered_count/original_len*100:.1f}%)")

    if BUILD_ALPHA_FEATURES:
        print("  Building alpha features...")
        df = build_alpha_features(
            df,
            market_returns=None,
            include_interactions=INCLUDE_INTERACTIONS,
        )

    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    ticker_artifacts = {
        "ticker": ticker,
        "horizons": {},
    }

    for target in TARGETS:
        horizon_short = target.replace("target_", "").replace("_return", "")
        print("\n" + "-" * 80)
        print(f"Target: {target} ({horizon_short})")
        print("-" * 80)

        print("  Creating horizon-specific features...")
        df_horizon = create_horizon_specific_features(df, horizon_short)

        feature_cols = [
            c for c in df_horizon.columns if c not in TARGETS and c != "ticker"
        ]

        X_raw = df_horizon[feature_cols]
        y_raw = df_horizon[target]

        # Drop rows where target is NaN
        valid_target_idx = y_raw.dropna().index

        # Drop rows where ALL features are NaN (keep rows with some valid features)
        valid_feature_idx = X_raw.dropna(how='all').index

        # Get intersection
        valid_idx = valid_target_idx.intersection(valid_feature_idx)

        if len(valid_idx) < 100:
            print(f"  [SKIP] Insufficient valid data after target filter: {len(valid_idx)} samples")
            continue

        # Now subset and drop only rows with NaN in any feature column
        X_subset = X_raw.loc[valid_idx]
        y_subset = y_raw.loc[valid_idx]

        # Drop rows with NaN in any feature
        final_valid_idx = X_subset.dropna().index

        if len(final_valid_idx) < 100:
            print(f"  [SKIP] Insufficient valid data after feature filter: {len(final_valid_idx)} samples")
            continue

        X = X_subset.loc[final_valid_idx]
        y = y_subset.loc[final_valid_idx]

        print(f"  Valid samples: {len(X)}")

        print("  Preprocessing targets...")
        y_processed, transform_params = preprocess_targets(
            y,
            method=TARGET_TRANSFORM,
            normalize=NORMALIZE_TARGETS,
            denoise=USE_KALMAN_FILTER,
            denoise_window=5,
        )

        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train = X.iloc[:train_end]
        y_train = y_processed.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y_processed.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_test = y_processed.iloc[val_end:]
        y_test_original = y.iloc[val_end:]

        print(f"  Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        print("  Scaling features...")
        X_train_scaled, scaler = robust_scale_features(X_train)
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

        horizon_artifacts = {
            "base_models": {},
            "val_predictions": {},
            "test_predictions": {},
            "metrics": {},
            "has_neural": False,
            "feature_columns": list(X_train.columns),
            "scaler": scaler,
            "transform_params": transform_params,
        }

        print("  Training base models...")
        gb_results = train_gradient_boosting_models(
            X_train_scaled, y_train, X_val_scaled, y_val
        )

        horizon_artifacts["base_models"].update(gb_results["models"])
        horizon_artifacts["val_predictions"].update(gb_results["val_preds"])

        for model_name, model in gb_results["models"].items():
            test_pred_scaled = model.predict(X_test_scaled)
            test_pred = inverse_preprocess_targets(test_pred_scaled, transform_params)

            horizon_artifacts["test_predictions"][model_name] = test_pred

            metrics = compute_metrics(y_test_original.values, test_pred)
            horizon_artifacts["metrics"][model_name] = metrics
            print_metrics(metrics, model_name, prefix="    ")

        if train_neural and ticker in NEURAL_TICKERS:
            print(f"\n  Neural models (lookback={LOOKBACK_WINDOW}):")

            X_train_seq, y_train_seq = create_sequence_dataset(
                X_train_scaled, y_train, LOOKBACK_WINDOW
            )
            X_val_seq, y_val_seq = create_sequence_dataset(
                X_val_scaled, y_val, LOOKBACK_WINDOW
            )
            X_test_seq, y_test_seq = create_sequence_dataset(
                X_test_scaled, y_test, LOOKBACK_WINDOW
            )

            n_features = X_train_scaled.shape[1]

            for neural_type in ["deep_lstm", "advanced_tcn"]:
                try:
                    model, val_preds = train_neural_model(
                        X_train_seq,
                        y_train_seq,
                        X_val_seq,
                        y_val_seq,
                        n_features,
                        model_type=neural_type,
                    )

                    if model is not None:
                        horizon_artifacts["base_models"][neural_type] = model
                        horizon_artifacts["val_predictions"][neural_type] = val_preds

                        model.eval()
                        test_loader = DataLoader(
                            TimeSeriesDataset(
                                torch.FloatTensor(X_test_seq),
                                torch.FloatTensor(y_test_seq),
                            ),
                            batch_size=64,
                            shuffle=False,
                            num_workers=0,
                        )

                        test_preds_list = []
                        with torch.no_grad():
                            for batch_X, _ in test_loader:
                                preds = model(batch_X)
                                test_preds_list.append(preds.cpu().numpy())

                        test_pred_scaled = np.concatenate(test_preds_list)
                        test_pred = inverse_preprocess_targets(
                            test_pred_scaled, transform_params
                        )

                        horizon_artifacts["test_predictions"][neural_type] = test_pred

                        y_test_seq_original = y_test_original.values[LOOKBACK_WINDOW - 1 :]

                        metrics = compute_metrics(y_test_seq_original, test_pred)
                        horizon_artifacts["metrics"][neural_type] = metrics
                        print_metrics(metrics, neural_type, prefix="    ")

                except Exception as e:
                    print(f"    [WARN] {neural_type} failed: {e}")

            horizon_artifacts["lookback_window"] = LOOKBACK_WINDOW
            horizon_artifacts["has_neural"] = True

        if USE_STACKING and len(horizon_artifacts["base_models"]) > 2:
            print("\n  Training stacking ensemble...")

            try:
                stacking_models = {
                    k: v
                    for k, v in horizon_artifacts["base_models"].items()
                    if k not in ["deep_lstm", "advanced_tcn", "transformer", "lstm", "tcn"]
                }

                ensemble = train_stacking_ensemble(
                    X_train_scaled,
                    y_train,
                    stacking_models,
                    n_folds=STACKING_FOLDS,
                    use_meta_features=USE_META_FEATURES,
                )

                horizon_artifacts["stacking_ensemble"] = ensemble

                ensemble_test_pred_scaled = predict_with_ensemble(
                    X_test_scaled, ensemble, return_components=False
                )
                ensemble_test_pred = inverse_preprocess_targets(
                    ensemble_test_pred_scaled, transform_params
                )

                horizon_artifacts["test_predictions"]["stacking_ensemble"] = (
                    ensemble_test_pred
                )

                metrics = compute_metrics(y_test_original.values, ensemble_test_pred)
                horizon_artifacts["metrics"]["stacking_ensemble"] = metrics
                print_metrics(metrics, "Stacking Ensemble", prefix="    ")

            except Exception as e:
                print(f"    [WARN] Stacking ensemble failed: {e}")

        ticker_artifacts["horizons"][target] = horizon_artifacts

    artifact_path = MODELS_DIR / f"{ticker}_numeric_ensemble.pkl"
    joblib.dump(ticker_artifacts, artifact_path)
    print(f"\n[SAVED] {artifact_path}")

    return ticker_artifacts


# ======================================================================
# MAIN
# ======================================================================


def main() -> None:
    print("=" * 80)
    print("NEXUSSIGNAL PHASE 3: NUMERIC MODEL TRAINING")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"  Train neural models: {TRAIN_NEURAL}")
    if TRAIN_NEURAL:
        print(f"  Neural tickers: {len(NEURAL_TICKERS)} / {len(TICKERS)}")
        print(f"  Lookback window: {LOOKBACK_WINDOW}")
    print(f"  Filter bad data: {FILTER_BAD_DATA}")
    print(f"  Build alpha features: {BUILD_ALPHA_FEATURES}")
    print(f"  Use stacking: {USE_STACKING}")
    print(f"  Target transform: {TARGET_TRANSFORM}")
    print(f"  Normalize targets: {NORMALIZE_TARGETS}")
    print(f"  Features directory: {FEATURES_DIR}")
    print(f"  Output directory: {MODELS_DIR}")

    available_tickers = []
    for ticker in TICKERS:
        if (FEATURES_DIR / f"{ticker}_1h.parquet").exists():
            available_tickers.append(ticker)

    print(f"\n  Available tickers: {len(available_tickers)} / {len(TICKERS)}")
    if available_tickers:
        print(f"  {', '.join(available_tickers)}")

    results = {}
    for ticker in available_tickers:
        train_neural_for_ticker = TRAIN_NEURAL and ticker in NEURAL_TICKERS
        result = train_ticker_models(ticker, train_neural=train_neural_for_ticker)
        if result is not None:
            results[ticker] = result

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    summary_rows = []
    for ticker, artifacts in results.items():
        row = {"Ticker": ticker}

        for target in TARGETS:
            if target not in artifacts["horizons"]:
                continue

            horizon_short = target.replace("target_", "").replace("_return", "")
            metrics = artifacts["horizons"][target]["metrics"].get(
                "stacking_ensemble",
                artifacts["horizons"][target]["metrics"].get("catboost", {}),
            )

            row[f"{horizon_short}_MAE"] = metrics.get("mae", np.nan)
            row[f"{horizon_short}_RMSE"] = metrics.get("rmse", np.nan)
            row[f"{horizon_short}_R2"] = metrics.get("r2", np.nan)
            row[f"{horizon_short}_DirAcc"] = metrics.get("directional_accuracy", np.nan)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    print("\nBest Model Performance Summary:")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("[WARN] No tickers successfully trained.")

    summary_path = MODELS_DIR / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVED] Summary: {summary_path}")
    print(f"\n[COMPLETE] Trained models for {len(results)} tickers")
    print(f"Artifacts saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
