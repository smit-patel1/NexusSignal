"""
Phase 3: Numeric Model Training & SOTA Ensemble Engine

Trains per-ticker ensemble models to predict:
- target_1h_return
- target_4h_return
- target_24h_return

Architecture:
- Group A: LightGBM, XGBoost, CatBoost (all tickers)
- Group B: LSTM, TCN (configurable subset)
- Ensemble: CatBoost meta-learner
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

# Gradient Boosting Models
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# PyTorch and PyTorch Lightning
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Scikit-learn
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.app.config import TICKERS
from scripts.utils.parquet_utils import read_parquet
from scripts.models.utils import (
    time_based_split,
    create_sequence_dataset,
    compute_metrics,
    print_metrics
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Neural model configuration
TRAIN_NEURAL = True  # Set to False to skip neural models (faster training)
NEURAL_TICKERS = TICKERS[:10]  # Train neural models only for first 10 tickers
LOOKBACK_WINDOW = 32  # Sequence length for LSTM/TCN

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
FEATURES_DIR = DATA_DIR / "processed" / "features"
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "numeric"

# Ensure output directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Target columns
TARGETS = ['target_1h_return', 'target_4h_return', 'target_24h_return']

# ============================================================================
# PYTORCH LIGHTNING MODELS
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Features array (samples, lookback, n_features)
            y: Targets array (samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMRegressor(pl.LightningModule):
    """LSTM-based regression model for time series."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, lookback, features)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out).squeeze()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_final = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove future information (causal convolution)
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        # Match sequence lengths
        if res.size(2) > out.size(2):
            res = res[:, :, :out.size(2)]
        elif res.size(2) < out.size(2):
            out = out[:, :, :res.size(2)]

        return self.relu_final(out + res)


class TCNRegressor(pl.LightningModule):
    """Temporal Convolutional Network for regression."""

    def __init__(
        self,
        n_features: int,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        super().__init__()
        self.save_hyperparameters()

        if num_channels is None:
            num_channels = [32, 64, 32]

        self.learning_rate = learning_rate

        # TCN layers
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_channels = n_features if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation = 2 ** i

            layers.append(TCNBlock(
                in_channels, out_channels, kernel_size, dilation, dropout
            ))

        self.tcn = nn.Sequential(*layers)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, lookback, features)
        # TCN expects: (batch, features, lookback)
        x = x.transpose(1, 2)

        tcn_out = self.tcn(x)
        # Take last timestep
        last_out = tcn_out[:, :, -1]

        return self.fc(last_out).squeeze()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_gradient_boosting_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, any]:
    """
    Train LightGBM, XGBoost, and CatBoost models.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets

    Returns:
        Dictionary of trained models and validation predictions
    """
    models = {}
    val_preds = {}

    # LightGBM
    print("    Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    models['lightgbm'] = lgb_model
    val_preds['lightgbm'] = lgb_model.predict(X_val)

    # XGBoost
    print("    Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    models['xgboost'] = xgb_model
    val_preds['xgboost'] = xgb_model.predict(X_val)

    # CatBoost
    print("    Training CatBoost...")
    cat_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50
    )
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    models['catboost'] = cat_model
    val_preds['catboost'] = cat_model.predict(X_val)

    return {'models': models, 'val_preds': val_preds}


def train_neural_model(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    n_features: int,
    model_type: str = 'lstm'
) -> Tuple[Optional[pl.LightningModule], Optional[np.ndarray]]:
    """
    Train LSTM or TCN model.

    Args:
        X_train_seq: Training sequences (samples, lookback, features)
        y_train_seq: Training targets
        X_val_seq: Validation sequences
        y_val_seq: Validation targets
        n_features: Number of features
        model_type: 'lstm' or 'tcn'

    Returns:
        Tuple of (trained model, validation predictions)
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Create model
    if model_type == 'lstm':
        print("    Training LSTM...")
        model = LSTMRegressor(n_features=n_features)
    elif model_type == 'tcn':
        print("    Training TCN...")
        model = TCNRegressor(n_features=n_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Get validation predictions
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            preds = model(batch_X)
            val_preds.append(preds.cpu().numpy())

    val_preds = np.concatenate(val_preds)

    return model, val_preds


def train_ensemble_blender(
    val_preds_dict: Dict[str, np.ndarray],
    y_val: np.ndarray
) -> CatBoostRegressor:
    """
    Train CatBoost meta-learner to blend base model predictions.

    Args:
        val_preds_dict: Dictionary of validation predictions from base models
        y_val: True validation targets

    Returns:
        Trained blender model
    """
    # Stack predictions as features
    stacked_preds = np.column_stack(list(val_preds_dict.values()))

    print(f"    Training ensemble blender with {len(val_preds_dict)} base models...")

    blender = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=4,
        random_seed=42,
        verbose=False
    )

    blender.fit(stacked_preds, y_val)

    return blender


def train_ticker_models(ticker: str, train_neural: bool = TRAIN_NEURAL) -> Optional[Dict]:
    """
    Train all models for a single ticker.

    Args:
        ticker: Stock ticker symbol
        train_neural: Whether to train neural models for this ticker

    Returns:
        Dictionary containing all trained models and metadata, or None if failed
    """
    print(f"\n{'=' * 80}")
    print(f"Training models for {ticker}")
    print(f"{'=' * 80}")

    # Load data
    feature_path = FEATURES_DIR / f"{ticker}_1h.parquet"
    if not feature_path.exists():
        print(f"[SKIP] No feature file found for {ticker}")
        return None

    df = read_parquet(feature_path)

    # Ensure datetime index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Sort by index
    df = df.sort_index()

    # Check for targets
    missing_targets = [t for t in TARGETS if t not in df.columns]
    if missing_targets:
        print(f"[SKIP] Missing targets: {missing_targets}")
        return None

    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Separate features and targets
    feature_cols = [col for col in df.columns if col not in TARGETS]
    X = df[feature_cols].copy()

    # Time-based split
    X_train, X_val, X_test = time_based_split(X)

    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Storage for all models and metrics
    ticker_artifacts = {
        'ticker': ticker,
        'feature_columns': feature_cols,
        'scaler': scaler,
        'horizons': {}
    }

    # Train for each horizon
    for target in TARGETS:
        print(f"\n{'-' * 80}")
        print(f"Target: {target}")
        print(f"{'-' * 80}")

        y_train, y_val, y_test = time_based_split(df[target])

        horizon_artifacts = {
            'base_models': {},
            'val_predictions': {},
            'test_predictions': {},
            'metrics': {}
        }

        # ====================================================================
        # GROUP A: Gradient Boosting Models
        # ====================================================================
        gb_results = train_gradient_boosting_models(
            X_train_scaled, y_train,
            X_val_scaled, y_val
        )

        horizon_artifacts['base_models'].update(gb_results['models'])
        horizon_artifacts['val_predictions'].update(gb_results['val_preds'])

        # Get test predictions
        for model_name, model in gb_results['models'].items():
            test_pred = model.predict(X_test_scaled)
            horizon_artifacts['test_predictions'][model_name] = test_pred

            # Compute metrics
            metrics = compute_metrics(y_test, test_pred)
            horizon_artifacts['metrics'][model_name] = metrics
            print_metrics(metrics, model_name, prefix="    ")

        # ====================================================================
        # GROUP B: Neural Models (if enabled for this ticker)
        # ====================================================================
        if train_neural and ticker in NEURAL_TICKERS:
            print(f"\n  Neural models (LOOKBACK={LOOKBACK_WINDOW}):")

            # Create sequences
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

            # Train LSTM
            lstm_model, lstm_val_pred = train_neural_model(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                n_features, model_type='lstm'
            )

            if lstm_model is not None:
                horizon_artifacts['base_models']['lstm'] = lstm_model
                horizon_artifacts['val_predictions']['lstm'] = lstm_val_pred

                # Test predictions
                lstm_model.eval()
                with torch.no_grad():
                    test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)
                    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
                    lstm_test_pred = []
                    for batch_X, _ in test_loader:
                        preds = lstm_model(batch_X)
                        lstm_test_pred.append(preds.cpu().numpy())
                    lstm_test_pred = np.concatenate(lstm_test_pred)

                horizon_artifacts['test_predictions']['lstm'] = lstm_test_pred

                # Metrics
                metrics = compute_metrics(y_test_seq, lstm_test_pred)
                horizon_artifacts['metrics']['lstm'] = metrics
                print_metrics(metrics, 'LSTM', prefix="    ")

            # Train TCN
            tcn_model, tcn_val_pred = train_neural_model(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                n_features, model_type='tcn'
            )

            if tcn_model is not None:
                horizon_artifacts['base_models']['tcn'] = tcn_model
                horizon_artifacts['val_predictions']['tcn'] = tcn_val_pred

                # Test predictions
                tcn_model.eval()
                with torch.no_grad():
                    test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)
                    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
                    tcn_test_pred = []
                    for batch_X, _ in test_loader:
                        preds = tcn_model(batch_X)
                        tcn_test_pred.append(preds.cpu().numpy())
                    tcn_test_pred = np.concatenate(tcn_test_pred)

                horizon_artifacts['test_predictions']['tcn'] = tcn_test_pred

                # Metrics
                metrics = compute_metrics(y_test_seq, tcn_test_pred)
                horizon_artifacts['metrics']['tcn'] = metrics
                print_metrics(metrics, 'TCN', prefix="    ")

            # Store sequence info
            horizon_artifacts['lookback_window'] = LOOKBACK_WINDOW
            horizon_artifacts['has_neural'] = True
        else:
            horizon_artifacts['has_neural'] = False

        # ====================================================================
        # ENSEMBLE: Train meta-learner
        # ====================================================================
        print(f"\n  Ensemble:")

        blender = train_ensemble_blender(
            horizon_artifacts['val_predictions'],
            y_val if not horizon_artifacts['has_neural'] else y_val_seq
        )

        horizon_artifacts['blender'] = blender

        # Ensemble test predictions
        test_preds_for_blender = []
        test_y = y_test

        for model_name in horizon_artifacts['val_predictions'].keys():
            test_preds_for_blender.append(horizon_artifacts['test_predictions'][model_name])

        # Align test targets if neural models were used
        if horizon_artifacts['has_neural']:
            test_y = y_test_seq

        stacked_test_preds = np.column_stack(test_preds_for_blender)
        ensemble_test_pred = blender.predict(stacked_test_preds)

        horizon_artifacts['test_predictions']['ensemble'] = ensemble_test_pred

        # Ensemble metrics
        ensemble_metrics = compute_metrics(test_y, ensemble_test_pred)
        horizon_artifacts['metrics']['ensemble'] = ensemble_metrics
        print_metrics(ensemble_metrics, 'Ensemble', prefix="    ")

        # Store horizon artifacts
        ticker_artifacts['horizons'][target] = horizon_artifacts

    # Save artifacts
    artifact_path = MODELS_DIR / f"{ticker}_numeric_ensemble.pkl"
    joblib.dump(ticker_artifacts, artifact_path)
    print(f"\n[SAVED] {artifact_path}")

    return ticker_artifacts


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("NEXUSSIGNAL PHASE 3: NUMERIC MODEL TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Train neural models: {TRAIN_NEURAL}")
    if TRAIN_NEURAL:
        print(f"  Neural tickers: {len(NEURAL_TICKERS)} / {len(TICKERS)}")
        print(f"  Lookback window: {LOOKBACK_WINDOW}")
    print(f"  Features directory: {FEATURES_DIR}")
    print(f"  Output directory: {MODELS_DIR}")

    # Discover available tickers
    available_tickers = []
    for ticker in TICKERS:
        feature_path = FEATURES_DIR / f"{ticker}_1h.parquet"
        if feature_path.exists():
            available_tickers.append(ticker)

    print(f"\n  Available tickers: {len(available_tickers)} / {len(TICKERS)}")
    print(f"  {', '.join(available_tickers)}")

    # Train models for each ticker
    results = {}
    for ticker in available_tickers:
        train_neural_for_ticker = TRAIN_NEURAL and ticker in NEURAL_TICKERS
        result = train_ticker_models(ticker, train_neural=train_neural_for_ticker)
        if result is not None:
            results[ticker] = result

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    summary_data = []

    for ticker, artifacts in results.items():
        row = {'Ticker': ticker}

        for target in TARGETS:
            horizon_short = target.replace('target_', '').replace('_return', '')
            metrics = artifacts['horizons'][target]['metrics'].get('ensemble', {})

            row[f'{horizon_short}_MAE'] = metrics.get('mae', np.nan)
            row[f'{horizon_short}_RMSE'] = metrics.get('rmse', np.nan)
            row[f'{horizon_short}_R2'] = metrics.get('r2', np.nan)
            row[f'{horizon_short}_DirAcc'] = metrics.get('directional_accuracy', np.nan)

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    print("\nEnsemble Performance Summary:")
    print(summary_df.to_string(index=False))

    # Save summary
    summary_path = MODELS_DIR / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVED] Summary: {summary_path}")

    print(f"\n[COMPLETE] Trained models for {len(results)} tickers")
    print(f"Artifacts saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
