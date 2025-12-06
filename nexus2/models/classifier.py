"""
Barrier Classification Models

Instead of regression, classify which barrier will be hit:
- Upper barrier (profit-taking)
- Lower barrier (stop-loss)
- Vertical barrier (time expiry)

This is a 3-class classification problem where outputs are:
- P(upper barrier hit | features)
- P(lower barrier hit | features)
- P(vertical barrier hit | features)

For meta-labeling, we also have a binary classifier:
- P(primary model is correct | features)
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p) = -α * (1-p)^γ * log(p)
    
    When an example is misclassified (p is small):
    - (1-p) is large, so the modulating factor is large
    - More weight on hard examples
    
    When correctly classified with high confidence (p is large):
    - (1-p) is small, so the modulating factor is small
    - Less weight on easy examples
    
    Reference: Lin et al. (2017), "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weight (or None for equal weights)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits (batch, n_classes)
            targets: Class indices (batch,)
        
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # p_t = softmax(logits)[target]
        
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BarrierClassifier(pl.LightningModule):
    """
    Neural network for Triple Barrier classification.
    
    Predicts probability of hitting each barrier type:
    - Class 0: Upper barrier (profit-taking)
    - Class 1: Lower barrier (stop-loss)
    - Class 2: Vertical barrier (time expiry)
    
    Output probabilities are calibrated using temperature scaling.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_classes: int = 3,
        hidden_layers: list = [256, 128, 64],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        class_weights: Optional[list] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0
    ):
        """
        Initialize barrier classifier.
        
        Args:
            input_dim: Number of input features
            n_classes: Number of classes (default 3 for TBM)
            hidden_layers: Hidden layer sizes
            dropout: Dropout rate
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            class_weights: Optional weights for imbalanced classes
            use_focal_loss: Whether to use focal loss
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Loss function
        if use_focal_loss:
            self.loss_fn = FocalLoss(gamma=focal_gamma)
        else:
            weight = torch.tensor(class_weights) if class_weights else None
            self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, input_dim)
        
        Returns:
            Logits (batch, n_classes)
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict calibrated probabilities.
        
        Uses temperature scaling for calibration.
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            # Temperature scaling
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=-1)
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.
        """
        probs = self.predict_proba(x)
        return probs.argmax(dim=-1)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        
        # Per-class accuracy
        for c in range(self.n_classes):
            mask = y == c
            if mask.sum() > 0:
                class_acc = (preds[mask] == c).float().mean()
                self.log(f'val_acc_class_{c}', class_acc)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
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
    
    def calibrate_temperature(
        self,
        val_loader: torch.utils.data.DataLoader,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> float:
        """
        Calibrate temperature parameter on validation set.
        
        Optimizes temperature to minimize NLL on validation set
        while keeping model weights frozen.
        
        Args:
            val_loader: Validation data loader
            lr: Learning rate for temperature optimization
            max_iter: Maximum iterations
        
        Returns:
            Calibrated temperature value
        """
        self.eval()
        
        # Collect all validation predictions
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                logits = self(x)
                all_logits.append(logits)
                all_targets.append(y)
        
        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        
        # Optimize temperature
        temperature = nn.Parameter(torch.ones(1, device=logits.device))
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        
        def eval_nll():
            optimizer.zero_grad()
            scaled_logits = logits / temperature
            loss = F.cross_entropy(scaled_logits, targets)
            loss.backward()
            return loss
        
        optimizer.step(eval_nll)
        
        # Update model temperature
        self.temperature.data = temperature.data
        
        return temperature.item()


def create_barrier_classifier(
    input_dim: int,
    config: Optional[dict] = None
) -> BarrierClassifier:
    """
    Factory function for creating barrier classifier.
    """
    if config is None:
        config = {}
    
    return BarrierClassifier(
        input_dim=input_dim,
        n_classes=config.get('n_classes', 3),
        hidden_layers=config.get('hidden_layers', [256, 128, 64]),
        dropout=config.get('dropout', 0.3),
        learning_rate=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5),
        use_focal_loss=config.get('use_focal_loss', True),
    )


class MetaClassifier(BarrierClassifier):
    """
    Binary classifier for meta-labeling.
    
    Predicts P(primary model is correct).
    
    Uses same architecture as BarrierClassifier but with:
    - 2 classes (correct / incorrect)
    - Additional features (primary model predictions)
    """
    
    def __init__(
        self,
        input_dim: int,
        **kwargs
    ):
        kwargs['n_classes'] = 2
        super().__init__(input_dim, **kwargs)
    
    def predict_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict confidence in primary model.
        
        Returns P(correct) for bet sizing.
        """
        probs = self.predict_proba(x)
        return probs[:, 1]  # P(class=1) = P(correct)

