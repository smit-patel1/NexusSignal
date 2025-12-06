# NexusSignal 2.0

**Production-Grade Quantitative Research System**

A complete redesign following modern Financial ML best practices from Lopez de Prado's *Advances in Financial Machine Learning*.

---

## What's New in v2.0?

NexusSignal 2.0 is a **ground-up rebuild** that fixes fundamental flaws in the v1 architecture:

| Component | v1 (Legacy) | v2.0 (New) |
|-----------|-------------|------------|
| **Labeling** | Fixed-horizon returns | Triple Barrier Method |
| **Features** | Technical indicators only | + Microstructure + Entropy + Regime |
| **Stationarity** | None | Fractional Differencing |
| **Validation** | Simple train/val/test | Combinatorial Purged CV |
| **Models** | Point regression (MSE) | Probabilistic classification |
| **Outputs** | Single prediction | Full distributions + confidence |
| **Signal Generation** | Raw predictions | Meta-labeling + position sizing |

---

## Quick Start

### 1. Install

```bash
# Install with ML dependencies
pip install -e ".[ml]"
```

### 2. Configure (Optional)

```bash
# Only needed if fetching new data
cp .env.example .env
# Edit .env with your API keys
```

### 3. Train

```bash
# Train all configured tickers
python -m nexus2.run_training

# Train single ticker
python -m nexus2.run_training --ticker AAPL

# Use custom config
python -m nexus2.run_training --config configs/my_config.yaml
```

### 4. Evaluate

```bash
# Evaluation and signal generation happens automatically during training
# Results are saved to outputs/results/
```

---

## Architecture Overview

```
NexusSignal/
├── nexus2/                    # NexusSignal 2.0 Core
│   ├── data/                  # Fractional differencing, sampling
│   ├── labeling/              # Triple barrier, meta-labeling
│   ├── features/              # Microstructure, entropy, regime
│   ├── models/                # Quantile NN, MDN, classifier
│   ├── validation/            # CPCV, financial metrics
│   ├── signals/               # Signal generation, position sizing
│   ├── pipeline/              # Training orchestrator
│   ├── utils/                 # Data loading, logging
│   └── config_default.yaml    # Default hyperparameters
│
├── data/                      # Data Storage
│   ├── raw/prices/            # Raw OHLCV (.parquet)
│   ├── raw/news/              # Raw news (.jsonl)
│   ├── processed/features/    # Legacy v1 features
│   └── interim/               # Intermediate outputs
│
├── outputs/                   # Training Outputs
│   ├── models/                # Trained checkpoints
│   ├── predictions/           # Prediction CSVs
│   ├── logs/                  # Training logs
│   └── results/               # Evaluation results
│
└── configs/                   # Custom configuration overrides
```

---

## Key Features

### 1. **Triple Barrier Method**
Instead of predicting raw returns, TBM defines *tradable outcomes*:
- Upper barrier → Profit target (label = 1)
- Lower barrier → Stop loss (label = -1)
- Vertical barrier → Time expiry (label based on sign)

### 2. **Fractional Differencing**
Makes time series stationary while preserving memory. Automatically finds optimal `d` parameter.

### 3. **Advanced Features**
- **Microstructure**: VPIN, Kyle's Lambda, Roll spread, order flow imbalance
- **Entropy**: ApEn, SampEn (market complexity measures)
- **Regime Detection**: HMM-based regime classification

### 4. **Combinatorial Purged Cross-Validation**
Prevents information leakage through:
- Purging: Remove overlapping labels between train/test
- Embargo: Additional time gap after test set
- Combinatorial: Multiple train/test group combinations

### 5. **Probabilistic Models**
- **Quantile Regression NN**: Predicts full distribution (quantiles)
- **Mixture Density Network**: Predicts mixture of Gaussians
- **Barrier Classifier**: Predicts probability of hitting barriers

### 6. **Meta-Labeling**
Two-stage pipeline:
1. Primary model: Predicts side (long/short)
2. Meta-model: Predicts probability of success
3. Filter: Only trade high-confidence signals

### 7. **Financial Metrics**
- Precision@K: Accuracy of top-K signals
- Brier Score: Probability calibration
- Deflated Sharpe Ratio: Adjusts for multiple testing
- Probabilistic Sharpe Ratio: P(true SR > benchmark)

---

## Programmatic Usage

```python
from nexus2.config import NexusConfig
from nexus2.pipeline.trainer import NexusSignalTrainer

# Load configuration
config = NexusConfig()

# Create trainer
trainer = NexusSignalTrainer(config)

# Train single ticker
result = trainer.train_ticker('AAPL')

# Access results
print(f"CV Accuracy: {result.primary_metrics['mean_cv_accuracy']:.2%}")
print(f"Sharpe Ratio: {result.signal_metrics['sharpe_ratio']:.2f}")
print(f"Precision@10: {result.signal_metrics['precision_at_10']:.2%}")
```

---

## Configuration

All hyperparameters are in `nexus2/config_default.yaml`. Key settings:

```yaml
# Triple Barrier Method
triple_barrier:
  profit_taking_multiplier: 2.0  # Upper barrier = 2.0 * volatility
  stop_loss_multiplier: 2.0      # Lower barrier = 2.0 * volatility
  max_holding_period: 24         # Maximum bars to hold

# Validation
validation:
  method: "cpcv"
  n_splits: 5
  n_test_groups: 2
  purge_length: 24
  embargo_pct: 0.01

# Model
model:
  type: "quantile_nn"  # or "mdn", "classifier"
  hidden_layers: [256, 128, 64]
  dropout: 0.3
  learning_rate: 0.001
```

Create custom configs in `configs/` and pass with `--config`.

---

## Data

The system uses existing data collected in `data/raw/`:
- **Prices**: 22 tickers × 2 timeframes (1h, daily)
- **News**: Daily news articles (optional for v2.0)

No need to re-fetch data. The new feature engineering will process existing raw data.

---

## Documentation

See **[nexus2/README.md](nexus2/README.md)** for detailed architecture documentation.

---

## References

1. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Lopez de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge.
3. Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
4. Easley, D. et al. (2012). "VPIN and the Flash Crash"
5. Kyle, A. (1985). "Continuous Auctions and Insider Trading"

---

## License

TBD

---

## Migration from v1

All legacy code (v1) has been removed. The new architecture is incompatible by design.

**What was removed:**
- `backend/` - Old API
- `scripts/` - Legacy training scripts
- `models/` - Old trained models
- `dashboard/` - Old dashboard
- All legacy feature engineering

**What was preserved:**
- `data/raw/` - All collected price and news data
- `data/processed/` - Legacy features (for reference only)

**To use v2.0:**
Simply run the new training pipeline. It will process raw data with the new feature engineering and labeling system.

---

## Status

**Current Phase:** NexusSignal 2.0 Implementation Complete ✅

All core components implemented:
- ✅ Data engineering (fractional differencing, sampling)
- ✅ Labeling system (TBM, meta-labeling)
- ✅ Feature engineering 2.0 (microstructure, entropy, regime)
- ✅ Probabilistic models (Quantile NN, MDN, classifier)
- ✅ Validation framework (CPCV, financial metrics)
- ✅ Signal generation (meta-labeling, position sizing)
- ✅ Training pipeline (end-to-end orchestrator)

**Next Steps:**
1. Run training on all tickers
2. Evaluate performance vs. v1
3. Deploy top-performing models
4. Build monitoring dashboard (optional)

---

## Support

For questions or issues, please refer to the documentation in `nexus2/README.md` or review the code directly.
