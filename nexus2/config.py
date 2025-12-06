"""
NexusSignal 2.0 Configuration System

Centralized configuration using Pydantic for validation and type safety.
All hyperparameters are externalized for experiment tracking.
"""

from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class FractionalDiffConfig(BaseModel):
    """Configuration for fractional differencing."""
    d_min: float = Field(0.0, description="Minimum differencing order to search")
    d_max: float = Field(1.0, description="Maximum differencing order to search")
    d_step: float = Field(0.01, description="Step size for d search")
    adf_threshold: float = Field(0.05, description="ADF p-value threshold for stationarity")
    weight_threshold: float = Field(1e-5, description="Minimum weight threshold for FFD")
    max_lag: int = Field(100, description="Maximum lag for weight computation")


class TripleBarrierConfig(BaseModel):
    """Configuration for Triple Barrier Method labeling."""
    # Barrier parameters
    profit_taking_multiplier: float = Field(2.0, description="Profit taking = multiplier * volatility")
    stop_loss_multiplier: float = Field(2.0, description="Stop loss = multiplier * volatility")
    max_holding_period: int = Field(24, description="Maximum holding period in bars")
    
    # Volatility estimation
    volatility_lookback: int = Field(20, description="Lookback for volatility estimation")
    volatility_method: Literal["std", "parkinson", "garman_klass", "yang_zhang"] = Field(
        "yang_zhang", description="Volatility estimation method"
    )
    
    # Minimum return threshold
    min_return_threshold: float = Field(0.0001, description="Minimum return to consider non-zero")
    
    # Side prediction
    use_side_prediction: bool = Field(True, description="Use side prediction for barriers")


class MetaLabelingConfig(BaseModel):
    """Configuration for meta-labeling."""
    enabled: bool = Field(True, description="Enable meta-labeling pipeline")
    min_confidence: float = Field(0.5, description="Minimum confidence for position sizing")
    bet_sizing_method: Literal["linear", "sigmoid", "kelly"] = Field(
        "sigmoid", description="Bet sizing method"
    )
    kelly_fraction: float = Field(0.25, description="Fraction of Kelly criterion to use")


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""
    # Microstructure features
    vpin_volume_buckets: int = Field(50, description="Number of volume buckets for VPIN")
    vpin_window: int = Field(20, description="Rolling window for VPIN")
    kyle_lambda_window: int = Field(20, description="Window for Kyle's Lambda")
    roll_spread_window: int = Field(20, description="Window for Roll spread")
    
    # Entropy features
    entropy_embedding_dim: int = Field(2, description="Embedding dimension for entropy")
    entropy_tolerance: float = Field(0.2, description="Tolerance for entropy calculation")
    entropy_window: int = Field(50, description="Window for rolling entropy")
    
    # Regime features
    hmm_n_regimes: int = Field(3, description="Number of HMM regimes")
    hmm_n_iter: int = Field(100, description="HMM training iterations")
    
    # Technical features
    technical_windows: List[int] = Field([5, 10, 20, 50], description="Technical indicator windows")
    
    # Fractional differencing
    apply_frac_diff: bool = Field(True, description="Apply fractional differencing")


class ModelConfig(BaseModel):
    """Configuration for model architectures."""
    model_type: Literal["quantile_nn", "mdn", "ensemble"] = Field(
        "quantile_nn", description="Primary model type"
    )
    
    # Quantile NN config
    quantiles: List[float] = Field(
        [0.05, 0.25, 0.5, 0.75, 0.95],
        description="Quantiles to predict"
    )
    hidden_layers: List[int] = Field([256, 128, 64], description="Hidden layer sizes")
    dropout: float = Field(0.3, description="Dropout rate")
    
    # MDN config
    n_mixtures: int = Field(5, description="Number of mixture components")
    
    # Training config
    learning_rate: float = Field(0.001, description="Initial learning rate")
    weight_decay: float = Field(1e-5, description="L2 regularization")
    batch_size: int = Field(256, description="Batch size")
    max_epochs: int = Field(100, description="Maximum training epochs")
    early_stopping_patience: int = Field(15, description="Early stopping patience")
    
    # Scheduler
    scheduler_factor: float = Field(0.5, description="LR scheduler factor")
    scheduler_patience: int = Field(5, description="LR scheduler patience")


class ValidationConfig(BaseModel):
    """Configuration for validation framework."""
    method: Literal["cpcv", "purged_kfold", "walk_forward"] = Field(
        "cpcv", description="Cross-validation method"
    )
    
    # CPCV parameters
    n_splits: int = Field(5, description="Number of CV splits")
    n_test_groups: int = Field(2, description="Number of test groups for CPCV")
    
    # Purging and embargo
    purge_length: int = Field(24, description="Purge length in bars (max horizon)")
    embargo_pct: float = Field(0.01, description="Embargo percentage")
    
    # Walk forward
    train_period: int = Field(252 * 5, description="Training period in bars (5 days)")
    test_period: int = Field(252, description="Test period in bars (1 day)")


class SignalConfig(BaseModel):
    """Configuration for signal generation."""
    min_probability: float = Field(0.55, description="Minimum probability for signal")
    position_sizing: Literal["equal", "volatility_scaled", "kelly"] = Field(
        "volatility_scaled", description="Position sizing method"
    )
    max_position: float = Field(1.0, description="Maximum position size (fraction of capital)")
    target_volatility: float = Field(0.15, description="Target annualized volatility")


class NexusConfig(BaseModel):
    """Master configuration for NexusSignal 2.0."""
    # Project paths
    data_dir: Path = Field(Path("data"), description="Data directory")
    models_dir: Path = Field(Path("models/nexus2"), description="Model output directory")
    results_dir: Path = Field(Path("results/nexus2"), description="Results directory")
    
    # Tickers and timeframes
    tickers: List[str] = Field(
        ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM"],
        description="Tickers to process"
    )
    base_timeframe: str = Field("1h", description="Base timeframe")
    
    # Component configs
    frac_diff: FractionalDiffConfig = Field(default_factory=FractionalDiffConfig)
    triple_barrier: TripleBarrierConfig = Field(default_factory=TripleBarrierConfig)
    meta_labeling: MetaLabelingConfig = Field(default_factory=MetaLabelingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    
    # Random seed
    seed: int = Field(42, description="Random seed for reproducibility")
    
    @classmethod
    def from_yaml(cls, path: Path) -> "NexusConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = NexusConfig()

