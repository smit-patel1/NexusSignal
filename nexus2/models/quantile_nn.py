"""
Quantile Regression Neural Network

Instead of predicting a single point estimate, Quantile NN predicts
multiple quantiles of the conditional distribution.

Key benefits:
- Full distributional output (not just mean)
- Captures uncertainty/heteroskedasticity
- Can estimate tail risk (VaR, CVaR)
- No distributional assumptions (non-parametric)

Loss function: Pinball loss (asymmetric L1)
    L_τ(y, ŷ) = τ * (y - ŷ)+ + (1-τ) * (ŷ - y)+
    
where (x)+ = max(0, x) and τ is the target quantile.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset


class QuantileLoss(nn.Module):
    """
    Pinball Loss for Quantile Regression.
    
    For quantile τ ∈ (0, 1):
    - If y > ŷ (underestimate): loss = τ * |y - ŷ|
    - If y < ŷ (overestimate): loss = (1-τ) * |y - ŷ|
    
    This asymmetric loss encourages the model to predict the τ-th quantile.
    
    Example:
        τ = 0.9 → Model penalized more for underestimating
        τ = 0.1 → Model penalized more for overestimating
    """
    
    def __init__(self, quantiles: List[float]):
        """
        Initialize with target quantiles.
        
        Args:
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
        """
        super().__init__()
        self.quantiles = quantiles
        self.register_buffer(
            'quantile_tensor',
            torch.tensor(quantiles, dtype=torch.float32)
        )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.
        
        Args:
            predictions: Shape (batch, n_quantiles)
            targets: Shape (batch,) or (batch, 1)
        
        Returns:
            Scalar loss averaged over batch and quantiles
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        
        # Errors for each quantile
        errors = targets - predictions  # (batch, n_quantiles)
        
        # Quantile-specific weights
        weights = torch.where(
            errors >= 0,
            self.quantile_tensor,
            self.quantile_tensor - 1
        )
        
        # Pinball loss
        loss = torch.abs(errors) * weights
        
        return loss.mean()


class QuantileRegressionNN(pl.LightningModule):
    """
    Neural network for multi-quantile regression.
    
    Architecture:
    - Input layer
    - Hidden layers with BatchNorm, ReLU, Dropout
    - Output layer with n_quantiles outputs
    
    The model outputs predictions for multiple quantiles simultaneously,
    providing a non-parametric estimate of the conditional distribution.
    
    Example:
        Quantiles [0.05, 0.25, 0.5, 0.75, 0.95] give:
        - Median prediction (0.5)
        - 50% confidence interval [0.25, 0.75]
        - 90% confidence interval [0.05, 0.95]
    """
    
    def __init__(
        self,
        input_dim: int,
        quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
        hidden_layers: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        use_batch_norm: bool = True
    ):
        """
        Initialize Quantile NN.
        
        Args:
            input_dim: Number of input features
            quantiles: Target quantiles to predict
            hidden_layers: Hidden layer sizes
            dropout: Dropout rate
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.quantiles = quantiles
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output head: one output per quantile
        self.output_layer = nn.Linear(prev_dim, len(quantiles))
        
        # Loss function
        self.loss_fn = QuantileLoss(quantiles)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, input_dim)
        
        Returns:
            Quantile predictions (batch, n_quantiles)
        """
        features = self.feature_extractor(x)
        quantile_preds = self.output_layer(features)
        return quantile_preds
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        
        # Also log coverage metrics
        self._log_coverage(y_hat, y)
        
        return loss
    
    def _log_coverage(self, y_hat: torch.Tensor, y: torch.Tensor):
        """Log quantile coverage metrics."""
        y = y.view(-1, 1)
        
        # Check if predictions are in the right order (monotonic)
        sorted_preds, _ = torch.sort(y_hat, dim=1)
        is_monotonic = (y_hat == sorted_preds).all()
        self.log('quantile_monotonic', float(is_monotonic))
        
        # Coverage: fraction of true values below each quantile
        for i, q in enumerate(self.quantiles):
            coverage = (y <= y_hat[:, i:i+1]).float().mean()
            self.log(f'coverage_q{int(q*100)}', coverage)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
    
    def predict_distribution(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get distribution predictions with derived quantities.
        
        Args:
            x: Input features
        
        Returns:
            Dictionary with:
            - quantiles: Raw quantile predictions
            - median: Median prediction (0.5 quantile)
            - mean: Approximate mean (average of quantiles)
            - iqr: Interquartile range (uncertainty)
            - lower_90: 5th percentile (downside risk)
            - upper_90: 95th percentile (upside potential)
        """
        self.eval()
        with torch.no_grad():
            q_preds = self(x)
        
        result = {'quantiles': q_preds}
        
        # Find indices for specific quantiles
        q_array = np.array(self.quantiles)
        
        if 0.5 in self.quantiles:
            median_idx = self.quantiles.index(0.5)
            result['median'] = q_preds[:, median_idx]
        else:
            result['median'] = q_preds.mean(dim=1)
        
        result['mean'] = q_preds.mean(dim=1)
        
        # IQR (uncertainty)
        if 0.25 in self.quantiles and 0.75 in self.quantiles:
            q25_idx = self.quantiles.index(0.25)
            q75_idx = self.quantiles.index(0.75)
            result['iqr'] = q_preds[:, q75_idx] - q_preds[:, q25_idx]
        
        # Tail quantiles
        if 0.05 in self.quantiles:
            result['lower_90'] = q_preds[:, self.quantiles.index(0.05)]
        if 0.95 in self.quantiles:
            result['upper_90'] = q_preds[:, self.quantiles.index(0.95)]
        
        return result


def create_quantile_model(
    input_dim: int,
    config: Optional[dict] = None
) -> QuantileRegressionNN:
    """
    Factory function for creating quantile model.
    
    Args:
        input_dim: Number of input features
        config: Optional configuration dictionary
    
    Returns:
        Configured QuantileRegressionNN
    """
    if config is None:
        config = {}
    
    return QuantileRegressionNN(
        input_dim=input_dim,
        quantiles=config.get('quantiles', [0.05, 0.25, 0.5, 0.75, 0.95]),
        hidden_layers=config.get('hidden_layers', [256, 128, 64]),
        dropout=config.get('dropout', 0.3),
        learning_rate=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5),
    )


class MonotonicQuantileNN(QuantileRegressionNN):
    """
    Quantile NN with monotonicity constraint.
    
    Ensures predicted quantiles are in order:
    q_0.05 < q_0.25 < q_0.5 < q_0.75 < q_0.95
    
    This prevents quantile crossing, which can occur in standard QR.
    
    Implementation: Use softplus on increments
        q_0 = base
        q_1 = q_0 + softplus(delta_1)
        q_2 = q_1 + softplus(delta_2)
        ...
    """
    
    def __init__(self, **kwargs):
        # Adjust output dim: we predict base + increments
        super().__init__(**kwargs)
        
        n_quantiles = len(self.quantiles)
        prev_dim = self.output_layer.in_features
        
        # Replace output layer with base + increments
        self.base_layer = nn.Linear(prev_dim, 1)  # Base prediction
        self.increment_layer = nn.Linear(prev_dim, n_quantiles - 1)  # Increments
        
        # Remove original output layer
        del self.output_layer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        
        # Base (lowest quantile)
        base = self.base_layer(features)  # (batch, 1)
        
        # Increments (guaranteed positive via softplus)
        increments = F.softplus(self.increment_layer(features))  # (batch, n_q - 1)
        
        # Cumulative sum for monotonicity
        quantiles = torch.cat([base, base + torch.cumsum(increments, dim=1)], dim=1)
        
        return quantiles

