"""
Test script for Phase 3 V2 setup validation.

Checks:
1. All required libraries are installed
2. Feature files are available
3. Advanced modules can be imported
4. Models can be instantiated
5. Basic pipeline can run on sample data
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all required libraries can be imported."""
    print("Testing imports...")

    required_imports = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "torch": "torch",
        "pytorch-lightning": "pytorch_lightning",
        "joblib": "joblib",
        "scipy": "scipy",
        "statsmodels": "statsmodels",
    }

    failed = []

    for display_name, module_name in required_imports.items():
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - pip install {display_name}")
            failed.append(display_name)

    if failed:
        print(f"\n  [ERROR] Missing packages: {', '.join(failed)}")
        return False

    return True


def test_advanced_modules():
    """Test that advanced modules can be imported."""
    print("\nTesting advanced modules...")

    modules = [
        ("advanced_preprocessing", "scripts.models.advanced_preprocessing"),
        ("advanced_features", "scripts.models.advanced_features"),
        ("advanced_validation", "scripts.models.advanced_validation"),
        ("advanced_neural_models", "scripts.models.advanced_neural_models"),
        ("advanced_ensemble", "scripts.models.advanced_ensemble"),
    ]

    failed = []

    for display_name, module_path in modules:
        try:
            __import__(module_path)
            print(f"  ✓ {display_name}")
        except Exception as e:
            print(f"  ✗ {display_name} - {e}")
            failed.append(display_name)

    if failed:
        print(f"\n  [ERROR] Failed to import: {', '.join(failed)}")
        return False

    return True


def test_data_availability():
    """Test that feature files are available."""
    print("\nTesting data availability...")

    from backend.app.config import TICKERS

    features_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "features"

    if not features_dir.exists():
        print(f"  ✗ Features directory not found: {features_dir}")
        print("     Run: python scripts/features/run_feature_build.py")
        return False

    feature_files = list(features_dir.glob("*_1h.parquet"))

    if not feature_files:
        print(f"  ✗ No feature files found in {features_dir}")
        print("     Run: python scripts/features/run_feature_build.py")
        return False

    print(f"  ✓ Found {len(feature_files)} feature files")

    # Test reading one file
    try:
        from scripts.utils.parquet_utils import read_parquet

        df = read_parquet(feature_files[0])
        print(f"  ✓ Sample file readable: {len(df)} rows, {len(df.columns)} columns")

        # Check for targets
        targets = ["target_1h_return", "target_4h_return", "target_24h_return"]
        missing = [t for t in targets if t not in df.columns]

        if missing:
            print(f"  ✗ Missing target columns: {missing}")
            return False

        print(f"  ✓ Target columns present")

    except Exception as e:
        print(f"  ✗ Error reading feature file: {e}")
        return False

    return True


def test_preprocessing_pipeline():
    """Test preprocessing functions on sample data."""
    print("\nTesting preprocessing pipeline...")

    try:
        import numpy as np
        import pandas as pd
        from scripts.models.advanced_preprocessing import (
            filter_bad_data,
            preprocess_targets,
            inverse_preprocess_targets,
            robust_scale_features,
        )

        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            "close": 100 + np.random.randn(100) * 5,
            "volume": np.random.randint(1000, 10000, 100),
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        })

        # Test bad data filtering
        df_filtered = filter_bad_data(df)
        print(f"  ✓ Bad data filtering: {len(df)} -> {len(df_filtered)} samples")

        # Test target preprocessing
        targets = pd.Series(np.random.randn(100) * 0.01)
        targets_processed, transform_params = preprocess_targets(
            targets, method="log", normalize=True
        )
        print(f"  ✓ Target preprocessing")

        # Test inverse transform
        targets_reconstructed = inverse_preprocess_targets(
            targets_processed.values, transform_params
        )
        print(f"  ✓ Inverse transform")

        # Test robust scaling
        features = df[["feature1", "feature2"]]
        scaled, scaler = robust_scale_features(features)
        print(f"  ✓ Robust scaling")

    except Exception as e:
        print(f"  ✗ Preprocessing pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_feature_engineering():
    """Test alpha feature engineering."""
    print("\nTesting alpha feature engineering...")

    try:
        import numpy as np
        import pandas as pd
        from scripts.models.advanced_features import (
            add_rolling_moments,
            add_rsi,
            add_macd,
            add_time_features,
        )

        # Create sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="h")
        df = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(200) * 0.5),
            "high": 100 + np.cumsum(np.random.randn(200) * 0.5) + 1,
            "low": 100 + np.cumsum(np.random.randn(200) * 0.5) - 1,
            "volume": np.random.randint(1000, 10000, 200),
        }, index=dates)

        # Test rolling moments
        df = add_rolling_moments(df, windows=[10, 20])
        print(f"  ✓ Rolling moments: added {len([c for c in df.columns if 'volatility' in c])} volatility features")

        # Test RSI
        df = add_rsi(df, periods=[14])
        print(f"  ✓ RSI: added {len([c for c in df.columns if 'rsi' in c])} RSI features")

        # Test MACD
        df = add_macd(df)
        print(f"  ✓ MACD: added {len([c for c in df.columns if 'macd' in c])} MACD features")

        # Test time features
        df = add_time_features(df)
        print(f"  ✓ Time features: added {len([c for c in df.columns if any(x in c for x in ['hour', 'dow', 'month'])])} time features")

    except Exception as e:
        print(f"  ✗ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_model_instantiation():
    """Test that models can be instantiated."""
    print("\nTesting model instantiation...")

    try:
        # Test neural models
        from scripts.models.advanced_neural_models import (
            DeepResidualLSTM,
            AdvancedTCN,
            TransformerRegressor,
        )

        model_lstm = DeepResidualLSTM(n_features=10)
        print("  ✓ DeepResidualLSTM")

        model_tcn = AdvancedTCN(n_features=10)
        print("  ✓ AdvancedTCN")

        model_transformer = TransformerRegressor(n_features=10)
        print("  ✓ TransformerRegressor")

        # Test ensemble models
        from scripts.models.advanced_ensemble import create_diverse_base_models

        base_models = create_diverse_base_models()
        print(f"  ✓ Base models: {len(base_models)} models created")

    except Exception as e:
        print(f"  ✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_validation_methods():
    """Test validation methods."""
    print("\nTesting validation methods...")

    try:
        import numpy as np
        import pandas as pd
        from scripts.models.advanced_validation import (
            TimeSeriesSplit,
            compute_fold_metrics,
        )

        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randn(100))

        # Test time series split
        tscv = TimeSeriesSplit(n_splits=3, strategy="expanding")
        splits = list(tscv.split(X))
        print(f"  ✓ TimeSeriesSplit: {len(splits)} splits generated")

        # Test metrics
        y_true = np.random.randn(50)
        y_pred = y_true + np.random.randn(50) * 0.1
        metrics = compute_fold_metrics(y_true, y_pred)
        print(f"  ✓ Fold metrics: {list(metrics.keys())}")

    except Exception as e:
        print(f"  ✗ Validation methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("PHASE 3 V2 SETUP VALIDATION")
    print("=" * 80)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Advanced Modules", test_advanced_modules()))
    results.append(("Data Availability", test_data_availability()))
    results.append(("Preprocessing Pipeline", test_preprocessing_pipeline()))
    results.append(("Feature Engineering", test_feature_engineering()))
    results.append(("Model Instantiation", test_model_instantiation()))
    results.append(("Validation Methods", test_validation_methods()))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n✓ All tests passed! Ready to train V2 models.")
        print("\nRun: python scripts/models/train_numeric_models_v2.py")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
