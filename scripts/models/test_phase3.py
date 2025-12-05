"""
Test script for Phase 3 setup validation.

Checks:
1. All required libraries are installed
2. Feature files are available
3. Training script can be imported
4. Inference module works
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all required libraries can be imported."""
    print("Testing imports...")

    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - pip install numpy")
        return False

    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - pip install pandas")
        return False

    try:
        import sklearn
        print("  ✓ scikit-learn")
    except ImportError:
        print("  ✗ scikit-learn - pip install scikit-learn")
        return False

    try:
        import lightgbm
        print("  ✓ lightgbm")
    except ImportError:
        print("  ✗ lightgbm - pip install lightgbm")
        return False

    try:
        import xgboost
        print("  ✓ xgboost")
    except ImportError:
        print("  ✗ xgboost - pip install xgboost")
        return False

    try:
        import catboost
        print("  ✓ catboost")
    except ImportError:
        print("  ✗ catboost - pip install catboost")
        return False

    try:
        import torch
        print("  ✓ torch")
    except ImportError:
        print("  ✗ torch - pip install torch")
        return False

    try:
        import pytorch_lightning as pl
        print("  ✓ pytorch-lightning")
    except ImportError:
        print("  ✗ pytorch-lightning - pip install pytorch-lightning")
        return False

    try:
        import joblib
        print("  ✓ joblib")
    except ImportError:
        print("  ✗ joblib - pip install joblib")
        return False

    return True


def test_data_availability():
    """Test that feature files are available."""
    print("\nTesting data availability...")

    features_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "features"

    if not features_dir.exists():
        print(f"  ✗ Features directory not found: {features_dir}")
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
        targets = ['target_1h_return', 'target_4h_return', 'target_24h_return']
        missing = [t for t in targets if t not in df.columns]

        if missing:
            print(f"  ✗ Missing target columns: {missing}")
            return False

        print(f"  ✓ Target columns present")

    except Exception as e:
        print(f"  ✗ Error reading feature file: {e}")
        return False

    return True


def test_training_script():
    """Test that training script can be imported."""
    print("\nTesting training script...")

    try:
        from scripts.models import train_numeric_models
        print("  ✓ Training script imports successfully")

        # Check key functions exist
        assert hasattr(train_numeric_models, 'train_ticker_models')
        print("  ✓ train_ticker_models function found")

        assert hasattr(train_numeric_models, 'train_gradient_boosting_models')
        print("  ✓ train_gradient_boosting_models function found")

        assert hasattr(train_numeric_models, 'train_ensemble_blender')
        print("  ✓ train_ensemble_blender function found")

    except Exception as e:
        print(f"  ✗ Error importing training script: {e}")
        return False

    return True


def test_inference_module():
    """Test that inference module works."""
    print("\nTesting inference module...")

    try:
        from backend.app.models import numeric_inference
        print("  ✓ Inference module imports successfully")

        # Check key functions exist
        assert hasattr(numeric_inference, 'load_numeric_model')
        print("  ✓ load_numeric_model function found")

        assert hasattr(numeric_inference, 'predict_numeric_returns')
        print("  ✓ predict_numeric_returns function found")

        assert hasattr(numeric_inference, 'get_model_info')
        print("  ✓ get_model_info function found")

    except Exception as e:
        print(f"  ✗ Error importing inference module: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("PHASE 3 SETUP VALIDATION")
    print("=" * 80)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Data Availability", test_data_availability()))
    results.append(("Training Script", test_training_script()))
    results.append(("Inference Module", test_inference_module()))

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
        print("\n✓ All tests passed! Ready to train models.")
        print("\nRun: python scripts/models/train_numeric_models.py")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
