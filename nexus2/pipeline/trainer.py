"""
NexusSignal 2.0 Training Pipeline

Complete training pipeline that orchestrates:
1. Data loading and feature building
2. Triple Barrier labeling
3. Model training with CPCV
4. Meta-labeling
5. Signal generation
6. Evaluation

This is the main entry point for training NexusSignal 2.0 models.
"""

from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
from dataclasses import dataclass
import json
import joblib
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from nexus2.config import NexusConfig
from nexus2.features.builder import FeatureBuilder, load_and_build_features
from nexus2.labeling.triple_barrier import TripleBarrierLabeler, BarrierConfig
from nexus2.labeling.meta_labeling import MetaLabeler, get_meta_labels
from nexus2.validation.cpcv import CombinatorialPurgedCV, PurgedKFold
from nexus2.validation.metrics import compute_all_metrics
from nexus2.models.classifier import BarrierClassifier, MetaClassifier, create_barrier_classifier
from nexus2.models.quantile_nn import QuantileRegressionNN, create_quantile_model
from nexus2.models.mdn import MixtureDensityNetwork, create_mdn_model
from nexus2.signals.generator import SignalGenerator


@dataclass
class TrainingResult:
    """Results from training a single ticker."""
    ticker: str
    primary_metrics: Dict[str, float]
    meta_metrics: Dict[str, float]
    signal_metrics: Dict[str, float]
    cv_scores: List[Dict[str, float]]
    feature_importance: Optional[pd.DataFrame]
    model_path: Path


class NexusTrainer:
    """
    Main training pipeline for NexusSignal 2.0.
    
    Complete workflow:
    1. Load raw data → build features
    2. Generate Triple Barrier labels
    3. Train primary model (barrier classifier)
    4. Generate meta-labels
    5. Train meta-model
    6. Evaluate with CPCV
    7. Generate signals
    8. Save artifacts
    
    Example:
        >>> config = NexusConfig.from_yaml('config.yaml')
        >>> trainer = NexusTrainer(config)
        >>> results = trainer.train_ticker('AAPL')
    """
    
    def __init__(self, config: NexusConfig):
        """
        Initialize trainer.
        
        Args:
            config: NexusConfig with all parameters
        """
        self.config = config
        config.ensure_dirs()
        
        # Initialize components
        self.feature_builder = FeatureBuilder(
            apply_frac_diff=config.features.apply_frac_diff,
            microstructure_windows=[10, 20, 50],
            entropy_windows=[20, 50, 100],
            n_regimes=config.features.hmm_n_regimes,
        )
        
        self.labeler = TripleBarrierLabeler(
            pt_multiplier=config.triple_barrier.profit_taking_multiplier,
            sl_multiplier=config.triple_barrier.stop_loss_multiplier,
            max_holding=config.triple_barrier.max_holding_period,
            vol_lookback=config.triple_barrier.volatility_lookback,
        )
        
        self.meta_labeler = MetaLabeler(
            min_confidence=config.meta_labeling.min_confidence,
            bet_sizing=config.meta_labeling.bet_sizing_method,
            kelly_fraction=config.meta_labeling.kelly_fraction,
        )
        
        self.signal_generator = SignalGenerator(
            threshold=config.signal.min_probability,
            position_method=config.signal.position_sizing,
            target_vol=config.signal.target_volatility,
            max_position=config.signal.max_position,
        )
        
        # Cross-validation
        self.cv = CombinatorialPurgedCV(
            n_splits=config.validation.n_splits,
            n_test_groups=config.validation.n_test_groups,
            purge_length=config.validation.purge_length,
            embargo_pct=config.validation.embargo_pct,
        )
        
        # Set random seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
    def train_ticker(
        self,
        ticker: str,
        save_artifacts: bool = True
    ) -> TrainingResult:
        """
        Train complete pipeline for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            save_artifacts: Whether to save model artifacts
        
        Returns:
            TrainingResult with metrics and paths
        """
        print(f"\n{'='*80}")
        print(f"Training NexusSignal 2.0 for {ticker}")
        print(f"{'='*80}")
        
        # Step 1: Load and build features
        print("\n[1/7] Building features...")
        try:
            features, feature_metadata = load_and_build_features(
                self.config.data_dir,
                ticker,
                self.feature_builder
            )
            print(f"  Built {feature_metadata['n_features']} features "
                  f"from {feature_metadata['n_samples']} samples")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            return None
        
        # Step 2: Generate labels
        print("\n[2/7] Generating Triple Barrier labels...")
        raw_path = self.config.data_dir / "raw" / "prices" / f"{ticker}_1h.parquet"
        df_raw = pd.read_parquet(raw_path)
        if 'timestamp' in df_raw.columns:
            df_raw = df_raw.set_index('timestamp')
        df_raw = df_raw.sort_index()
        
        # Standardize columns
        column_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        df_raw = df_raw.rename(columns=column_map)
        
        labels = self.labeler.fit_transform(df_raw)
        label_dist = self.labeler.get_label_distribution(labels)
        print(f"  Labels: {label_dist['pct_positive']:.1%} pos, "
              f"{label_dist['pct_negative']:.1%} neg, "
              f"{label_dist['pct_neutral']:.1%} neutral")
        print(f"  Total events: {label_dist['total_events']}")
        
        # Step 3: Align features and labels
        print("\n[3/7] Aligning features and labels...")
        common_idx = features.index.intersection(labels.index)
        X = features.loc[common_idx]
        y_labels = labels.loc[common_idx]
        
        # Convert to classification target (0=upper, 1=lower, 2=vertical)
        y_class = y_labels['label'].map({1: 0, -1: 1, 0: 2}).values
        
        print(f"  Aligned {len(X)} samples")
        
        # Step 4: Cross-validated training
        print("\n[4/7] Training with Combinatorial Purged CV...")
        cv_results = []
        all_test_preds = []
        all_test_targets = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv.split(X, y_labels)):
            print(f"  Fold {fold_idx + 1}/{self.cv.get_n_splits()}...", end=' ')
            
            X_train = X.loc[train_idx]
            X_test = X.loc[test_idx]
            y_train = y_class[X.index.isin(train_idx)]
            y_test = y_class[X.index.isin(test_idx)]
            
            # Train classifier
            model = self._train_classifier(
                X_train.values, y_train,
                X_test.values, y_test
            )
            
            # Evaluate
            proba = model.predict_proba(torch.tensor(X_test.values, dtype=torch.float32))
            preds = proba.numpy().argmax(axis=1)
            
            acc = (preds == y_test).mean()
            cv_results.append({'fold': fold_idx, 'accuracy': acc})
            print(f"Acc: {acc:.3f}")
            
            all_test_preds.extend(preds)
            all_test_targets.extend(y_test)
        
        mean_acc = np.mean([r['accuracy'] for r in cv_results])
        print(f"  Mean CV Accuracy: {mean_acc:.3f}")
        
        # Step 5: Train final model on all data
        print("\n[5/7] Training final model on full data...")
        final_model = self._train_classifier(X.values, y_class, X.values, y_class)
        
        # Step 6: Meta-labeling (simplified - use classifier confidence as meta)
        print("\n[6/7] Computing meta-labeling metrics...")
        final_proba = final_model.predict_proba(torch.tensor(X.values, dtype=torch.float32)).numpy()
        
        # Use max probability as confidence
        confidence = final_proba.max(axis=1)
        predicted_side = np.where(final_proba[:, 0] > final_proba[:, 1], 1, -1)
        
        # Meta metrics
        high_conf_mask = confidence > 0.6
        if high_conf_mask.sum() > 0:
            high_conf_acc = (np.where(final_proba[high_conf_mask].argmax(axis=1) == 0, 1, -1) == 
                           y_labels.loc[X.index[high_conf_mask], 'label'].values).mean()
        else:
            high_conf_acc = 0.0
        
        meta_metrics = {
            'mean_confidence': confidence.mean(),
            'high_conf_pct': high_conf_mask.mean(),
            'high_conf_accuracy': high_conf_acc,
        }
        print(f"  Mean confidence: {meta_metrics['mean_confidence']:.3f}")
        print(f"  High confidence accuracy: {high_conf_acc:.3f}")
        
        # Step 7: Signal evaluation
        print("\n[7/7] Evaluating signals...")
        signals_df = self.signal_generator.generate(
            final_proba,
            meta_confidence=confidence,
            volatility=X['volatility'].values if 'volatility' in X.columns else None
        )
        
        # Compute returns
        actual_returns = y_labels.loc[X.index, 'return'].values
        signal_metrics = self.signal_generator.evaluate_signals(
            signals_df, actual_returns, y_labels.loc[X.index, 'label'].values
        )
        
        print(f"  Hit rate: {signal_metrics.get('hit_rate', 0):.2%}")
        print(f"  Sharpe ratio: {signal_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Trade frequency: {signal_metrics.get('trade_frequency', 0):.2%}")
        
        # Save artifacts
        model_path = self.config.models_dir / f"{ticker}_nexus2.pkl"
        if save_artifacts:
            print(f"\n[SAVE] Saving to {model_path}")
            self._save_artifacts(
                model_path, ticker, final_model, 
                self.feature_builder, self.labeler,
                feature_metadata, cv_results
            )
        
        return TrainingResult(
            ticker=ticker,
            primary_metrics={'mean_cv_accuracy': mean_acc},
            meta_metrics=meta_metrics,
            signal_metrics=signal_metrics,
            cv_scores=cv_results,
            feature_importance=None,  # Could extract from model
            model_path=model_path
        )
    
    def _train_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> BarrierClassifier:
        """Train barrier classifier."""
        model = create_barrier_classifier(
            input_dim=X_train.shape[1],
            config={
                'n_classes': 3,
                'hidden_layers': self.config.model.hidden_layers,
                'dropout': self.config.model.dropout,
                'learning_rate': self.config.model.learning_rate,
                'weight_decay': self.config.model.weight_decay,
            }
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.model.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.model.batch_size)
        
        # Train
        trainer = pl.Trainer(
            max_epochs=self.config.model.max_epochs,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.model.early_stopping_patience,
                    mode='min'
                )
            ],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator='auto',
            devices=1,
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        return model
    
    def _save_artifacts(
        self,
        path: Path,
        ticker: str,
        model: BarrierClassifier,
        feature_builder: FeatureBuilder,
        labeler: TripleBarrierLabeler,
        feature_metadata: Dict,
        cv_results: List[Dict]
    ):
        """Save all training artifacts."""
        artifacts = {
            'ticker': ticker,
            'model_state': model.state_dict(),
            'model_config': model.hparams,
            'd_values': feature_builder.d_values_,
            'hmm_result': feature_builder.hmm_detector_.get_regime_stats() if feature_builder.hmm_detector_ else None,
            'feature_metadata': feature_metadata,
            'labeler_config': {
                'pt_multiplier': labeler.pt_multiplier,
                'sl_multiplier': labeler.sl_multiplier,
                'max_holding': labeler.max_holding,
                'vol_lookback': labeler.vol_lookback,
            },
            'cv_results': cv_results,
            'config': self.config.model_dump(),
        }
        
        joblib.dump(artifacts, path)
    
    def train_all(self) -> Dict[str, TrainingResult]:
        """Train models for all configured tickers."""
        results = {}
        
        for ticker in self.config.tickers:
            try:
                result = self.train_ticker(ticker)
                if result is not None:
                    results[ticker] = result
            except Exception as e:
                print(f"\n[ERROR] Failed to train {ticker}: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        summary = []
        for ticker, result in results.items():
            summary.append({
                'ticker': ticker,
                'cv_accuracy': result.primary_metrics['mean_cv_accuracy'],
                'hit_rate': result.signal_metrics.get('hit_rate', 0),
                'sharpe': result.signal_metrics.get('sharpe_ratio', 0),
            })
        
        if summary:
            summary_df = pd.DataFrame(summary)
            print(summary_df.to_string(index=False))
            
            # Save summary
            summary_path = self.config.results_dir / "training_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\n[SAVED] Summary: {summary_path}")
        
        return results


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NexusSignal 2.0")
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--ticker', type=str, default=None, help='Single ticker to train')
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = NexusConfig.from_yaml(Path(args.config))
    else:
        config = NexusConfig()
    
    # Train
    trainer = NexusTrainer(config)
    
    if args.ticker:
        trainer.train_ticker(args.ticker)
    else:
        trainer.train_all()


if __name__ == "__main__":
    main()

