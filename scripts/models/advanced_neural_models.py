"""
Advanced neural network architectures for time series prediction.

Includes:
- Deep LSTM with residual connections
- Advanced TCN with attention
- Transformer encoder
- Layer normalization and attention pooling
- Better learning rate scheduling
"""

from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence data."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)

        Returns:
            Weighted sum: (batch, hidden_size)
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        weighted = (x * attn_weights).sum(dim=1)  # (batch, hidden_size)

        return weighted


class ResidualLSTMBlock(nn.Module):
    """LSTM block with residual connection and layer norm."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        # Projection for residual if dimensions don't match
        self.projection = (
            nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            (batch, seq_len, hidden_size)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Residual connection
        if self.projection is not None:
            residual = self.projection(x)
        else:
            residual = x

        # Add residual and normalize
        out = self.layer_norm(lstm_out + residual)

        return out


class DeepResidualLSTM(pl.LightningModule):
    """
    Deep LSTM with residual connections, layer norm, and attention pooling.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list = [128, 128, 64],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        use_attention: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_attention = use_attention

        # Stack of residual LSTM blocks
        self.lstm_blocks = nn.ModuleList()

        input_size = n_features
        for hidden_size in hidden_sizes:
            block = ResidualLSTMBlock(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=dropout,
            )
            self.lstm_blocks.append(block)
            input_size = hidden_size

        # Pooling
        if use_attention:
            self.pooling = AttentionPooling(hidden_sizes[-1])
        else:
            self.pooling = None

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.LayerNorm(hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[-1] // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch,) predictions
        """
        # Pass through residual LSTM blocks
        out = x
        for block in self.lstm_blocks:
            out = block(out)

        # Pool sequence
        if self.use_attention:
            out = self.pooling(out)
        else:
            out = out[:, -1, :]  # Last timestep

        # Output
        out = self.fc(out).squeeze(-1)

        return out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class ResidualTCNBlock(nn.Module):
    """TCN block with residual connection and layer norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.ln1 = nn.LayerNorm(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.ln2 = nn.LayerNorm(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)

        Returns:
            (batch, out_channels, seq_len)
        """
        # First conv
        out = self.conv1(x)
        out = out[:, :, : -self.conv1.padding[0]]  # Remove padding
        out = out.transpose(1, 2)  # (batch, seq_len, channels) for LayerNorm
        out = self.ln1(out)
        out = out.transpose(1, 2)  # Back to (batch, channels, seq_len)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv
        out = self.conv2(out)
        out = out[:, :, : -self.conv2.padding[0]]  # Remove padding
        out = out.transpose(1, 2)
        out = self.ln2(out)
        out = out.transpose(1, 2)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)

        # Match sequence length
        min_len = min(res.size(2), out.size(2))
        res = res[:, :, :min_len]
        out = out[:, :, :min_len]

        return self.final_relu(out + res)


class AdvancedTCN(pl.LightningModule):
    """
    Advanced TCN with residual connections, layer norm, and attention.
    """

    def __init__(
        self,
        n_features: int,
        num_channels: list = [64, 128, 128, 64],
        kernel_size: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        use_attention: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_attention = use_attention

        # TCN blocks
        layers = []
        for i, out_channels in enumerate(num_channels):
            in_channels = n_features if i == 0 else num_channels[i - 1]
            dilation = 2**i

            block = ResidualTCNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            layers.append(block)

        self.tcn = nn.Sequential(*layers)

        # Pooling
        if use_attention:
            self.pooling = AttentionPooling(num_channels[-1])
        else:
            self.pooling = None

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.LayerNorm(num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch,) predictions
        """
        # Transpose for conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)

        # TCN
        out = self.tcn(x)

        # Back to (batch, seq_len, channels)
        out = out.transpose(1, 2)

        # Pool
        if self.use_attention:
            out = self.pooling(out)
        else:
            out = out[:, -1, :]

        # Output
        out = self.fc(out).squeeze(-1)

        return out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class TransformerRegressor(pl.LightningModule):
    """
    Transformer encoder for time series regression.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch,) predictions
        """
        # Project to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer
        x = self.transformer(x)

        # Use last timestep
        x = x[:, -1, :]

        # Output
        out = self.fc(x).squeeze(-1)

        return out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# Import numpy for positional encoding
import numpy as np
