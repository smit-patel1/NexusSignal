# Quick Start: Upgraded Training Pipeline (V2)

## Installation

```bash
# Install upgraded dependencies
pip install -e ".[ml]"

# Verify scipy and statsmodels are installed
python -c "import scipy; import statsmodels; print('✓ All dependencies installed')"
```

## Step 1: Run V2 Training

```bash
# Train upgraded models (will take longer than v1 due to additional models and features)
python scripts/models/train_numeric_models_v2.py
```

**Expected time:**
- Without neural models (~15-30 min for 22 tickers)
- With neural models (~60-90 min for 10 neural + 12 baseline tickers)

## Step 2: Compare Results

```bash
# Generate comparison report
python scripts/models/compare_v1_v2.py
```

This will show:
- Aggregate statistics (mean R², MAE, Dir Acc)
- Per-ticker improvements
- Top 5 best improvements
- Any regressions

## Step 3: Review Results

### Check Training Summary

```python
import pandas as pd

# Load v2 results
summary = pd.read_csv("models/numeric_v2/training_summary_v2.csv")

# View best performing tickers
print(summary.sort_values("1h_R2", ascending=False).head(10))

# Check directional accuracy
print(summary[["Ticker", "1h_DirAcc", "4h_DirAcc", "24h_DirAcc"]])
```

### Load and Inspect Model

```python
import joblib

# Load trained model
artifact = joblib.load("models/numeric_v2/AAPL_numeric_ensemble_v2.pkl")

# Check what's inside
print("Horizons:", list(artifact["horizons"].keys()))

# Check 1h horizon
horizon_1h = artifact["horizons"]["target_1h_return"]
print("Base models:", list(horizon_1h["base_models"].keys()))
print("Metrics:", horizon_1h["metrics"])

# Check if stacking ensemble was trained
if "stacking_ensemble" in horizon_1h:
    print("✓ Stacking ensemble available")
    print("Metrics:", horizon_1h["metrics"]["stacking_ensemble"])
```

## Configuration Tuning

### If Results Are Still Poor:

**1. Enable Kalman Filtering** (for noisy data):
```python
# In train_numeric_models_v2.py
USE_KALMAN_FILTER = True
```

**2. Try Different Target Transform**:
```python
# Options: 'log', 'arctanh', 'winsorize'
TARGET_TRANSFORM = "arctanh"  # Better for extreme values
```

**3. Increase Stacking Folds**:
```python
STACKING_FOLDS = 7  # More robust cross-validation
```

**4. Stricter Data Filtering**:
```python
MIN_VOLUME_PERCENTILE = 5.0  # Filter more low-volume periods
MAX_PRICE_JUMP = 10.0  # Stricter outlier removal
```

### If Training Is Too Slow:

**1. Disable Neural Models**:
```python
TRAIN_NEURAL = False
```

**2. Reduce Stacking Folds**:
```python
STACKING_FOLDS = 3
```

**3. Disable Interaction Features**:
```python
INCLUDE_INTERACTIONS = False
```

**4. Remove Slowest Models**:
```python
# In advanced_ensemble.py -> create_diverse_base_models()
# Comment out RandomForest (slowest)
# "random_forest": RandomForestRegressor(...),  # REMOVE THIS LINE
```

### If Memory Issues:

**1. Reduce Neural Ticker Count**:
```python
NEURAL_TICKERS = TICKERS[:5]  # Only top 5
```

**2. Reduce Lookback Window**:
```python
LOOKBACK_WINDOW = 16  # Smaller sequences
```

**3. Disable Meta-Features**:
```python
USE_META_FEATURES = False
```

## Understanding the Output

### Model Files

```
models/numeric_v2/
├── AAPL_numeric_ensemble_v2.pkl    # Per-ticker artifacts
├── MSFT_numeric_ensemble_v2.pkl
├── ...
├── training_summary_v2.csv         # Performance table
└── v1_v2_comparison.csv            # Comparison (after running compare script)
```

### Artifact Structure

```python
{
    "ticker": "AAPL",
    "horizons": {
        "target_1h_return": {
            "base_models": {
                "ridge": model,
                "lasso": model,
                "elasticnet": model,
                "random_forest": model,
                "hist_gbm": model,
                "lightgbm": model,
                "xgboost": model,
                "catboost": model,
                "deep_lstm": model (if neural enabled),
                "advanced_tcn": model (if neural enabled),
            },
            "stacking_ensemble": {
                "base_models": {...},
                "meta_learner": CatBoostRegressor,
                "use_meta_features": bool,
            },
            "scaler": RobustScaler,
            "transform_params": {
                "method": "log",
                "mean": float,
                "std": float,
            },
            "feature_columns": [...],
            "metrics": {
                "ridge": {...},
                "catboost": {...},
                "stacking_ensemble": {...},
            },
        },
        # Same for target_4h_return and target_24h_return
    }
}
```

### Interpreting Metrics

**R² (Coefficient of Determination):**
- **> 0.5**: Excellent (explains >50% of variance)
- **0.3 - 0.5**: Good (meaningful predictive power)
- **0.1 - 0.3**: Moderate (some signal)
- **< 0.1**: Weak (minimal predictive power)
- **< 0**: Worse than naive baseline (bad)

**Directional Accuracy:**
- **> 60%**: Excellent (strong directional edge)
- **55-60%**: Good (clear edge over random)
- **52-55%**: Moderate (slight edge)
- **50-52%**: Weak (barely better than random)
- **< 50%**: Bad (worse than coin flip)

**MAE (Mean Absolute Error):**
- Measures average prediction error in return units
- Lower is better
- For 1h returns: < 0.002 is excellent, < 0.004 is good
- For 24h returns: < 0.015 is excellent, < 0.030 is good

## Making Predictions

### Basic Prediction (Single Model)

```python
import joblib
import pandas as pd
from scripts.models.advanced_preprocessing import inverse_preprocess_targets

# Load model
artifact = joblib.load("models/numeric_v2/AAPL_numeric_ensemble_v2.pkl")
horizon_1h = artifact["horizons"]["target_1h_return"]

# Prepare features (must match training columns exactly)
feature_cols = horizon_1h["feature_columns"]
feature_row = pd.DataFrame({col: [value] for col, value in zip(feature_cols, feature_values)})

# Scale
scaler = horizon_1h["scaler"]
feature_scaled = scaler.transform(feature_row)
feature_scaled_df = pd.DataFrame(feature_scaled, columns=feature_cols)

# Predict with best single model (e.g., CatBoost)
catboost_model = horizon_1h["base_models"]["catboost"]
pred_scaled = catboost_model.predict(feature_scaled_df)

# Inverse transform
transform_params = horizon_1h["transform_params"]
pred = inverse_preprocess_targets(pred_scaled, transform_params)

print(f"Predicted 1h return: {pred[0]:.4f}")
```

### Ensemble Prediction (Recommended)

```python
from scripts.models.advanced_ensemble import predict_with_ensemble

# Use stacking ensemble
ensemble = horizon_1h["stacking_ensemble"]
pred_scaled, components = predict_with_ensemble(
    feature_scaled_df, ensemble, return_components=True
)

# Inverse transform
pred = inverse_preprocess_targets(pred_scaled, transform_params)

print(f"Ensemble prediction: {pred[0]:.4f}")
print(f"Component predictions:")
for model_name, pred_value in components.items():
    print(f"  {model_name}: {pred_value[0]:.4f}")
```

## Troubleshooting

### Error: "No module named 'statsmodels'"

```bash
pip install statsmodels>=0.14.0
```

### Error: "No module named 'scipy'"

```bash
pip install scipy>=1.11.0
```

### Error: "CUDA out of memory" (for neural models)

```python
# Reduce batch size in train_neural_model()
train_loader = DataLoader(train_dataset, batch_size=32, ...)  # Was 64
```

### Warning: "Stacking ensemble failed"

This usually means:
- Not enough valid base models (need at least 3)
- Cross-validation failed on small datasets
- Memory issues

**Solution**: Check that at least 3 base models trained successfully, or disable stacking:
```python
USE_STACKING = False
```

### Poor Performance on Specific Ticker

1. Check data quality: `df.describe()`, look for NaNs
2. Increase minimum sample requirement
3. Enable Kalman filtering for that ticker
4. Check if ticker has sufficient history (>1000 samples recommended)

## Next Steps

1. **Backtest the models**: Implement walk-forward backtesting to validate performance
2. **Online learning**: Implement incremental updates as new data arrives
3. **Ensemble selection**: Automatically select best model per ticker
4. **Feature importance**: Analyze which features drive predictions
5. **Risk management**: Add position sizing based on prediction confidence

## Support

For issues or questions:
1. Check [PHASE3_V2_README.md](PHASE3_V2_README.md) for detailed documentation
2. Review error messages carefully - they often indicate configuration issues
3. Try with a single ticker first to isolate problems
4. Reduce complexity (disable neural, reduce folds) and gradually increase
