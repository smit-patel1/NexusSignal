# Phase 3: Numeric Model Training & SOTA Ensemble Engine

## Overview

Phase 3 implements a state-of-the-art ensemble training pipeline for predicting stock returns at multiple horizons:
- **1-hour returns** (short-term)
- **4-hour returns** (medium-term)
- **24-hour returns** (long-term)

## Architecture

### Per-Ticker Ensemble Strategy

Each ticker has its own ensemble of models trained independently:

**Group A: Gradient Boosting Models** (All Tickers)
- **LightGBM** - Fast gradient boosting with leaf-wise tree growth
- **XGBoost** - Extreme gradient boosting with regularization
- **CatBoost** - Categorical boosting with ordered target encoding

**Group B: Neural Time-Series Models** (Configurable Subset)
- **LSTM** - Long Short-Term Memory for temporal dependencies
- **TCN** - Temporal Convolutional Network for efficient sequence modeling

**Meta-Learner: CatBoost Blender**
- Learns optimal non-linear combinations of base model predictions
- Trained on validation set predictions
- Superior to linear blending methods

## Configuration

Edit `scripts/models/train_numeric_models.py`:

```python
# Neural model configuration
TRAIN_NEURAL = True  # Set to False to skip neural models (faster)
NEURAL_TICKERS = TICKERS[:10]  # Train neural only for first 10 tickers
LOOKBACK_WINDOW = 32  # Sequence length for LSTM/TCN
```

## Data Requirements

**Input:** Engineered features from Phase 2
- Location: `data/processed/features/{TICKER}_1h.parquet`
- Each file contains ~80-100 features + 3 target columns
- Targets: `target_1h_return`, `target_4h_return`, `target_24h_return`

**Output:** Trained ensemble artifacts
- Location: `models/numeric/{TICKER}_numeric_ensemble.pkl`
- Contains: All base models, blender, scaler, metadata

## Training Pipeline

### 1. Data Preparation

```python
# Time-based split (no shuffling)
Train: 70% (earliest data)
Validation: 15% (middle data)
Test: 15% (most recent data)

# Feature scaling
StandardScaler fit on training data only
```

### 2. Model Training

**For each ticker and each horizon:**

1. **Train Group A** (LightGBM, XGBoost, CatBoost)
   - Production-ready hyperparameters
   - Early stopping on validation set
   - Individual model evaluation

2. **Train Group B** (LSTM, TCN) - if enabled
   - Create sliding window sequences (lookback=32)
   - PyTorch Lightning training with early stopping
   - GPU acceleration if available

3. **Train Ensemble Blender**
   - Stack validation predictions from all base models
   - Train CatBoost meta-learner
   - Learn optimal model weights

4. **Evaluate on Test Set**
   - MAE, RMSE, R², Directional Accuracy
   - Compare individual models vs ensemble

### 3. Model Persistence

Each ticker's artifact contains:
```python
{
    'ticker': str,
    'feature_columns': list,  # Exact order for inference
    'scaler': StandardScaler,
    'horizons': {
        'target_1h_return': {
            'base_models': {
                'lightgbm': model,
                'xgboost': model,
                'catboost': model,
                'lstm': model or None,
                'tcn': model or None
            },
            'blender': CatBoostRegressor,
            'metrics': dict,
            'has_neural': bool,
            'lookback_window': int
        },
        # ... same for 4h and 24h
    }
}
```

## Usage

### Training Models

**Install ML dependencies:**
```bash
pip install -e ".[ml]"
```

**Train all models:**
```bash
python scripts/models/train_numeric_models.py
```

**Expected output:**
```
================================================================================
NEXUSSIGNAL PHASE 3: NUMERIC MODEL TRAINING
================================================================================

Configuration:
  Train neural models: True
  Neural tickers: 10 / 22
  Lookback window: 32
  ...

================================================================================
Training models for AAPL
================================================================================
Loaded 3487 samples with 85 columns
Split: Train=2440, Val=522, Test=523

Target: target_1h_return
--------------------------------------------------------------------------------
    Training LightGBM...
    Training XGBoost...
    Training CatBoost...
    LightGBM:
      MAE: 0.002156
      RMSE: 0.003842
      R²: 0.3421
      Directional Accuracy: 54.32%
    ...

  Neural models (LOOKBACK=32):
    Training LSTM...
    Training TCN...
    ...

  Ensemble:
    Training ensemble blender with 5 base models...
    Ensemble:
      MAE: 0.001987
      RMSE: 0.003654
      R²: 0.3856
      Directional Accuracy: 55.67%

[SAVED] models/numeric/AAPL_numeric_ensemble.pkl
```

### Inference

**Load and predict:**
```python
from backend.app.models.numeric_inference import (
    predict_numeric_returns,
    get_model_info,
    list_available_models
)

# List available models
tickers = list_available_models()
# ['AAPL', 'MSFT', 'NVDA', ...]

# Get model info
info = get_model_info('AAPL')
# {
#     'ticker': 'AAPL',
#     'feature_count': 85,
#     'has_neural': True,
#     'horizons': ['target_1h_return', 'target_4h_return', 'target_24h_return'],
#     'base_models': ['catboost', 'lightgbm', 'lstm', 'tcn', 'xgboost']
# }

# Make prediction
import pandas as pd

# Create feature row (must have exact columns from training)
feature_row = pd.Series({
    'close': 185.5,
    'sma_10': 184.2,
    'rsi_14': 58.3,
    # ... all 85 features
})

predictions = predict_numeric_returns('AAPL', feature_row)
# {
#     'target_1h_return': 0.0023,
#     'target_4h_return': 0.0048,
#     'target_24h_return': 0.0112,
#     'components': {
#         'lightgbm': {'1h': 0.0021, '4h': 0.0045, '24h': 0.0109},
#         'xgboost': {'1h': 0.0024, '4h': 0.0051, '24h': 0.0115},
#         'catboost': {'1h': 0.0022, '4h': 0.0047, '24h': 0.0111},
#         'lstm': None,  # Requires full lookback window
#         'tcn': None,
#         'ensemble': {'1h': 0.0023, '4h': 0.0048, '24h': 0.0112}
#     }
# }
```

## Performance Metrics

### Evaluation Metrics

**MAE (Mean Absolute Error)**
- Average absolute difference between predicted and actual returns
- Lower is better
- Interpretable in return units (e.g., 0.002 = 0.2% average error)

**RMSE (Root Mean Squared Error)**
- Square root of average squared error
- Penalizes large errors more than MAE
- Lower is better

**R² (Coefficient of Determination)**
- Proportion of variance explained by the model
- Range: -∞ to 1.0 (1.0 = perfect predictions)
- Higher is better

**Directional Accuracy**
- Percentage of correct sign predictions (up/down)
- Critical for trading strategies
- >50% indicates predictive power above random

### Expected Performance

**Gradient Boosting Models:**
- R² range: 0.25 - 0.45 (depending on horizon and ticker)
- Directional accuracy: 52% - 58%
- Best for: Capturing non-linear feature interactions

**Neural Models:**
- R² range: 0.20 - 0.40
- Directional accuracy: 51% - 56%
- Best for: Long-term temporal patterns

**Ensemble:**
- R² range: 0.30 - 0.50
- Directional accuracy: 54% - 60%
- Best overall: Combines strengths of all models

## File Structure

```
scripts/models/
├── __init__.py
├── train_numeric_models.py    # Main training pipeline
└── utils.py                    # Utilities (splitting, metrics)

backend/app/models/
├── __init__.py
└── numeric_inference.py        # Inference API

models/numeric/
├── AAPL_numeric_ensemble.pkl   # Trained artifacts
├── MSFT_numeric_ensemble.pkl
├── ...
└── training_summary.csv        # Performance summary
```

## Training Time Estimates

**Gradient Boosting Only** (TRAIN_NEURAL=False)
- Per ticker: ~30-60 seconds
- 22 tickers: ~15-25 minutes
- Recommended for: Quick iterations, Colab free tier

**With Neural Models** (TRAIN_NEURAL=True, 10 tickers)
- Per ticker (neural): ~3-5 minutes
- Per ticker (boosting only): ~30-60 seconds
- Total: ~45-70 minutes
- Recommended for: Full SOTA performance, Colab Pro

## Colab Setup

```python
# Install dependencies
!pip install lightgbm xgboost catboost torch pytorch-lightning scikit-learn joblib

# Clone repo
!git clone https://github.com/yourusername/NexusSignal.git
%cd NexusSignal

# Upload pre-built features (35 MB)
# Use Colab file upload: data/processed/features/

# Train models
!python scripts/models/train_numeric_models.py
```

## Troubleshooting

**Out of Memory:**
- Reduce `NEURAL_TICKERS` to fewer tickers
- Set `TRAIN_NEURAL = False`
- Reduce `LOOKBACK_WINDOW` to 16 or 24

**Slow Training:**
- Disable neural models: `TRAIN_NEURAL = False`
- Reduce gradient boosting `n_estimators` in training script
- Use Colab GPU runtime for neural models

**Poor Performance:**
- Check feature quality in Phase 2 outputs
- Ensure sufficient data (>2000 samples recommended)
- Verify no data leakage (features from future timestamps)

## Next Steps

**Phase 4: API Integration**
- Add FastAPI endpoints for model predictions
- Real-time feature computation from live prices
- Streaming predictions via WebSocket

**Phase 5: Backtesting & Signals**
- Historical performance simulation
- Signal generation from ensemble predictions
- Risk-adjusted portfolio optimization

**Phase 6: Advanced Ensembling**
- Multi-ticker meta-learning
- Transfer learning between similar stocks
- Online learning for model updates

## References

**Model Papers:**
- LightGBM: "A Highly Efficient Gradient Boosting Decision Tree" (Ke et al., 2017)
- XGBoost: "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)
- CatBoost: "CatBoost: unbiased boosting with categorical features" (Prokhorenkova et al., 2018)
- LSTM: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- TCN: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (Bai et al., 2018)

**Ensemble Learning:**
- "Ensemble Methods: Foundations and Algorithms" (Zhou, 2012)
- "Stacked Generalization" (Wolpert, 1992)
