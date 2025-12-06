"""
Mixture Density Network (MDN)

MDN outputs parameters of a mixture of Gaussians instead of a single
prediction. This allows modeling:
- Multi-modal distributions
- Heteroskedasticity (input-dependent variance)
- Asymmetric distributions

Output for each input:
- K mixture weights (π_1, ..., π_K) that sum to 1
- K means (μ_1, ..., μ_K)
- K standard deviations (σ_1, ..., σ_K)

The predicted distribution is:
    p(y|x) = Σ_k π_k * N(y | μ_k, σ_k²)

Reference: Bishop (1994), "Mixture Density Networks"
"""

from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MDNLayer(nn.Module):
    """
    MDN output layer that produces mixture parameters.
    
    From hidden features, outputs:
    - π: Mixture weights (K values, sum to 1)
    - μ: Component means (K values)
    - σ: Component stds (K values, positive)
    """
    
    def __init__(self, input_dim: int, n_mixtures: int):
        """
        Initialize MDN layer.
        
        Args:
            input_dim: Input feature dimension
            n_mixtures: Number of mixture components
        """
        super().__init__()
        
        self.n_mixtures = n_mixtures
        
        # Output layers
        self.pi_layer = nn.Linear(input_dim, n_mixtures)
        self.mu_layer = nn.Linear(input_dim, n_mixtures)
        self.sigma_layer = nn.Linear(input_dim, n_mixtures)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mixture parameters.
        
        Args:
            x: Hidden features (batch, input_dim)
        
        Returns:
            Tuple of (pi, mu, sigma):
            - pi: Mixture weights (batch, K), sum to 1
            - mu: Means (batch, K)
            - sigma: Stds (batch, K), positive
        """
        # Mixture weights (softmax ensures sum = 1)
        pi = F.softmax(self.pi_layer(x), dim=-1)
        
        # Means (unconstrained)
        mu = self.mu_layer(x)
        
        # Standard deviations (ELU + 1 ensures positive, more stable than exp)
        sigma = F.elu(self.sigma_layer(x)) + 1 + 1e-6
        
        return pi, mu, sigma


class MDNLoss(nn.Module):
    """
    Negative log-likelihood loss for MDN.
    
    For mixture distribution:
        p(y|x) = Σ_k π_k * N(y | μ_k, σ_k²)
        
    NLL = -log(p(y|x))
        = -log(Σ_k π_k * N(y | μ_k, σ_k²))
        
    Uses log-sum-exp trick for numerical stability.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood.
        
        Args:
            pi: Mixture weights (batch, K)
            mu: Means (batch, K)
            sigma: Stds (batch, K)
            target: True values (batch,) or (batch, 1)
        
        Returns:
            NLL loss
        """
        if target.dim() == 1:
            target = target.unsqueeze(-1)  # (batch, 1)
        
        # Gaussian log-likelihood for each component
        # log N(y | μ, σ²) = -0.5 * log(2π) - log(σ) - 0.5 * ((y-μ)/σ)²
        log_prob = (
            -0.5 * np.log(2 * np.pi)
            - torch.log(sigma)
            - 0.5 * ((target - mu) / sigma) ** 2
        )  # (batch, K)
        
        # Add log mixture weights
        log_weighted = torch.log(pi + 1e-10) + log_prob  # (batch, K)
        
        # Log-sum-exp for mixture
        nll = -torch.logsumexp(log_weighted, dim=-1)  # (batch,)
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class MixtureDensityNetwork(pl.LightningModule):
    """
    Full Mixture Density Network.
    
    Architecture:
    - Shared feature extractor (MLP)
    - MDN output layer (produces π, μ, σ)
    
    Capabilities:
    - Multi-modal predictions (e.g., price could go up OR down sharply)
    - Uncertainty estimation via mixture variance
    - Input-dependent confidence
    
    Example use case:
        Around earnings announcements, returns are often bimodal
        (large move up OR down). MDN can capture this.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_mixtures: int = 5,
        hidden_layers: list = [256, 128, 64],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize MDN.
        
        Args:
            input_dim: Number of input features
            n_mixtures: Number of Gaussian components
            hidden_layers: Hidden layer sizes
            dropout: Dropout rate
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.n_mixtures = n_mixtures
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Feature extractor
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # MDN output layer
        self.mdn_layer = MDNLayer(prev_dim, n_mixtures)
        
        # Loss
        self.loss_fn = MDNLoss()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, input_dim)
        
        Returns:
            Tuple of (pi, mu, sigma)
        """
        features = self.feature_extractor(x)
        pi, mu, sigma = self.mdn_layer(features)
        return pi, mu, sigma
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        pi, mu, sigma = self(x)
        loss = self.loss_fn(pi, mu, sigma, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        pi, mu, sigma = self(x)
        loss = self.loss_fn(pi, mu, sigma, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
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
    
    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict mixture mean (weighted average of component means).
        
        E[Y|X] = Σ_k π_k * μ_k
        """
        self.eval()
        with torch.no_grad():
            pi, mu, sigma = self(x)
        
        # Weighted mean
        return (pi * mu).sum(dim=-1)
    
    def predict_mode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict mixture mode (mean of most likely component).
        """
        self.eval()
        with torch.no_grad():
            pi, mu, sigma = self(x)
        
        # Most likely component
        max_idx = pi.argmax(dim=-1)
        
        # Gather corresponding mean
        return mu[torch.arange(len(mu)), max_idx]
    
    def predict_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict mixture variance (law of total variance).
        
        Var[Y|X] = E[σ²] + E[(μ - E[Y])²]
                 = Σ_k π_k * σ_k² + Σ_k π_k * (μ_k - μ_mixture)²
        """
        self.eval()
        with torch.no_grad():
            pi, mu, sigma = self(x)
        
        # Mixture mean
        mixture_mean = (pi * mu).sum(dim=-1, keepdim=True)
        
        # Within-component variance
        within_var = (pi * sigma ** 2).sum(dim=-1)
        
        # Between-component variance
        between_var = (pi * (mu - mixture_mean) ** 2).sum(dim=-1)
        
        return within_var + between_var
    
    def predict_quantile(
        self,
        x: torch.Tensor,
        quantile: float = 0.5,
        n_samples: int = 1000
    ) -> torch.Tensor:
        """
        Estimate quantile via sampling.
        
        No closed form for mixture quantiles, so we sample.
        """
        self.eval()
        with torch.no_grad():
            pi, mu, sigma = self(x)
        
        batch_size = x.shape[0]
        
        # Sample from mixture
        # 1. Sample component indices
        component_idx = torch.multinomial(pi, n_samples, replacement=True)  # (batch, n_samples)
        
        # 2. Sample from chosen Gaussians
        mu_selected = torch.gather(mu.unsqueeze(1).expand(-1, n_samples, -1), 2, component_idx.unsqueeze(-1)).squeeze(-1)
        sigma_selected = torch.gather(sigma.unsqueeze(1).expand(-1, n_samples, -1), 2, component_idx.unsqueeze(-1)).squeeze(-1)
        
        samples = mu_selected + sigma_selected * torch.randn_like(mu_selected)  # (batch, n_samples)
        
        # Compute quantile
        q_idx = int(quantile * n_samples)
        sorted_samples, _ = torch.sort(samples, dim=1)
        
        return sorted_samples[:, q_idx]
    
    def predict_distribution(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get full distribution predictions.
        
        Returns:
            Dictionary with mixture parameters and derived quantities
        """
        self.eval()
        with torch.no_grad():
            pi, mu, sigma = self(x)
        
        return {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'mean': (pi * mu).sum(dim=-1),
            'variance': self.predict_variance(x),
            'std': torch.sqrt(self.predict_variance(x)),
            'mode': self.predict_mode(x),
            'q05': self.predict_quantile(x, 0.05),
            'q50': self.predict_quantile(x, 0.50),
            'q95': self.predict_quantile(x, 0.95),
        }
    
    def sample(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Sample from predicted distribution.
        
        Args:
            x: Input features
            n_samples: Number of samples per input
        
        Returns:
            Samples (batch, n_samples)
        """
        self.eval()
        with torch.no_grad():
            pi, mu, sigma = self(x)
        
        # Sample component indices
        component_idx = torch.multinomial(pi, n_samples, replacement=True)
        
        # Gather parameters
        mu_selected = torch.gather(mu.unsqueeze(1).expand(-1, n_samples, -1), 2, component_idx.unsqueeze(-1)).squeeze(-1)
        sigma_selected = torch.gather(sigma.unsqueeze(1).expand(-1, n_samples, -1), 2, component_idx.unsqueeze(-1)).squeeze(-1)
        
        # Sample
        samples = mu_selected + sigma_selected * torch.randn_like(mu_selected)
        
        return samples


def create_mdn_model(
    input_dim: int,
    config: Optional[dict] = None
) -> MixtureDensityNetwork:
    """
    Factory function for creating MDN.
    
    Args:
        input_dim: Number of input features
        config: Optional configuration dictionary
    
    Returns:
        Configured MixtureDensityNetwork
    """
    if config is None:
        config = {}
    
    return MixtureDensityNetwork(
        input_dim=input_dim,
        n_mixtures=config.get('n_mixtures', 5),
        hidden_layers=config.get('hidden_layers', [256, 128, 64]),
        dropout=config.get('dropout', 0.3),
        learning_rate=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5),
    )

