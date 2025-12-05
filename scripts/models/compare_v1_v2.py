"""
Compare performance between v1 and v2 models.

Loads training summaries from both versions and generates comparison report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

# Paths
MODELS_V1 = Path(__file__).parent.parent.parent / "models" / "numeric"
MODELS_V2 = Path(__file__).parent.parent.parent / "models" / "numeric_v2"

SUMMARY_V1 = MODELS_V1 / "training_summary.csv"
SUMMARY_V2 = MODELS_V2 / "training_summary_v2.csv"


def load_summaries():
    """Load both summary files."""
    if not SUMMARY_V1.exists():
        print(f"[ERROR] V1 summary not found: {SUMMARY_V1}")
        return None, None

    if not SUMMARY_V2.exists():
        print(f"[ERROR] V2 summary not found: {SUMMARY_V2}")
        return None, None

    v1 = pd.read_csv(SUMMARY_V1)
    v2 = pd.read_csv(SUMMARY_V2)

    return v1, v2


def compute_improvements(v1, v2):
    """
    Compute improvement metrics between v1 and v2.

    Args:
        v1, v2: DataFrames with training summaries

    Returns:
        DataFrame with improvements
    """
    # Merge on ticker
    merged = v1.merge(v2, on="Ticker", suffixes=("_v1", "_v2"))

    improvements = pd.DataFrame({"Ticker": merged["Ticker"]})

    # For each horizon
    for horizon in ["1h", "4h", "24h"]:
        # R² improvement (higher is better)
        r2_v1 = merged[f"{horizon}_R2_v1"]
        r2_v2 = merged[f"{horizon}_R2_v2"]
        improvements[f"{horizon}_R2_delta"] = r2_v2 - r2_v1
        improvements[f"{horizon}_R2_pct_change"] = (
            (r2_v2 - r2_v1) / np.abs(r2_v1).clip(lower=0.01) * 100
        )

        # MAE improvement (lower is better)
        mae_v1 = merged[f"{horizon}_MAE_v1"]
        mae_v2 = merged[f"{horizon}_MAE_v2"]
        improvements[f"{horizon}_MAE_delta"] = mae_v2 - mae_v1
        improvements[f"{horizon}_MAE_pct_change"] = (
            (mae_v2 - mae_v1) / mae_v1 * 100
        )

        # Directional accuracy improvement
        dir_v1 = merged[f"{horizon}_DirAcc_v1"]
        dir_v2 = merged[f"{horizon}_DirAcc_v2"]
        improvements[f"{horizon}_DirAcc_delta"] = dir_v2 - dir_v1

    return improvements


def print_summary_statistics(v1, v2, improvements):
    """Print aggregate statistics."""
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    for horizon in ["1h", "4h", "24h"]:
        print(f"\n{horizon.upper()} Horizon:")
        print("-" * 40)

        # V1 stats
        r2_v1_mean = v1[f"{horizon}_R2"].mean()
        mae_v1_mean = v1[f"{horizon}_MAE"].mean()
        dir_v1_mean = v1[f"{horizon}_DirAcc"].mean()

        # V2 stats
        r2_v2_mean = v2[f"{horizon}_R2"].mean()
        mae_v2_mean = v2[f"{horizon}_MAE"].mean()
        dir_v2_mean = v2[f"{horizon}_DirAcc"].mean()

        print(f"  R² Score:")
        print(f"    V1: {r2_v1_mean:.4f}")
        print(f"    V2: {r2_v2_mean:.4f}")
        print(f"    Δ:  {r2_v2_mean - r2_v1_mean:+.4f}")

        print(f"  MAE:")
        print(f"    V1: {mae_v1_mean:.6f}")
        print(f"    V2: {mae_v2_mean:.6f}")
        print(f"    Δ:  {mae_v2_mean - mae_v1_mean:+.6f} ({(mae_v2_mean - mae_v1_mean) / mae_v1_mean * 100:+.1f}%)")

        print(f"  Directional Accuracy:")
        print(f"    V1: {dir_v1_mean:.2%}")
        print(f"    V2: {dir_v2_mean:.2%}")
        print(f"    Δ:  {dir_v2_mean - dir_v1_mean:+.2%}")

        # Count improvements
        r2_improved = (improvements[f"{horizon}_R2_delta"] > 0).sum()
        mae_improved = (improvements[f"{horizon}_MAE_delta"] < 0).sum()
        dir_improved = (improvements[f"{horizon}_DirAcc_delta"] > 0).sum()

        total_tickers = len(improvements)

        print(f"\n  Tickers Improved:")
        print(f"    R²: {r2_improved}/{total_tickers} ({r2_improved/total_tickers*100:.1f}%)")
        print(f"    MAE: {mae_improved}/{total_tickers} ({mae_improved/total_tickers*100:.1f}%)")
        print(f"    Dir Acc: {dir_improved}/{total_tickers} ({dir_improved/total_tickers*100:.1f}%)")


def identify_best_improvements(improvements):
    """Identify tickers with biggest improvements."""
    print("\n" + "=" * 80)
    print("TOP 5 IMPROVEMENTS (by R² delta)")
    print("=" * 80)

    for horizon in ["1h", "4h", "24h"]:
        print(f"\n{horizon.upper()} Horizon:")
        top_5 = improvements.nlargest(5, f"{horizon}_R2_delta")[
            ["Ticker", f"{horizon}_R2_delta", f"{horizon}_MAE_pct_change", f"{horizon}_DirAcc_delta"]
        ]

        print(top_5.to_string(index=False))


def identify_regressions(improvements):
    """Identify tickers where performance got worse."""
    print("\n" + "=" * 80)
    print("REGRESSIONS (where v2 performed worse)")
    print("=" * 80)

    for horizon in ["1h", "4h", "24h"]:
        regressions = improvements[improvements[f"{horizon}_R2_delta"] < 0]

        if len(regressions) > 0:
            print(f"\n{horizon.upper()} Horizon: {len(regressions)} tickers regressed")
            print(
                regressions[
                    ["Ticker", f"{horizon}_R2_delta", f"{horizon}_MAE_pct_change"]
                ].to_string(index=False)
            )
        else:
            print(f"\n{horizon.upper()} Horizon: No regressions ✓")


def main():
    print("=" * 80)
    print("V1 vs V2 COMPARISON REPORT")
    print("=" * 80)

    # Load summaries
    v1, v2 = load_summaries()

    if v1 is None or v2 is None:
        print("\n[ERROR] Cannot proceed without both summary files.")
        print("\nMake sure you have trained both v1 and v2 models:")
        print("  V1: python scripts/models/train_numeric_models.py")
        print("  V2: python scripts/models/train_numeric_models_v2.py")
        return

    print(f"\nLoaded summaries:")
    print(f"  V1: {len(v1)} tickers")
    print(f"  V2: {len(v2)} tickers")

    # Compute improvements
    improvements = compute_improvements(v1, v2)

    # Print statistics
    print_summary_statistics(v1, v2, improvements)

    # Best improvements
    identify_best_improvements(improvements)

    # Regressions
    identify_regressions(improvements)

    # Overall verdict
    print("\n" + "=" * 80)
    print("OVERALL VERDICT")
    print("=" * 80)

    avg_r2_improvement = improvements[[f"{h}_R2_delta" for h in ["1h", "4h", "24h"]]].mean().mean()
    avg_dir_improvement = improvements[[f"{h}_DirAcc_delta" for h in ["1h", "4h", "24h"]]].mean().mean()

    print(f"\nAverage R² improvement: {avg_r2_improvement:+.4f}")
    print(f"Average Dir Acc improvement: {avg_dir_improvement:+.2%}")

    if avg_r2_improvement > 0.1 and avg_dir_improvement > 0.03:
        print("\n✓ SUBSTANTIAL IMPROVEMENT - V2 is significantly better")
    elif avg_r2_improvement > 0.05 and avg_dir_improvement > 0.01:
        print("\n✓ MODERATE IMPROVEMENT - V2 shows measurable gains")
    elif avg_r2_improvement > 0:
        print("\n≈ SLIGHT IMPROVEMENT - V2 is marginally better")
    else:
        print("\n✗ NO IMPROVEMENT - V2 did not outperform V1")
        print("\nPossible reasons:")
        print("  - Insufficient data quality")
        print("  - Overfitting in cross-validation")
        print("  - Need to tune hyperparameters")
        print("  - Consider enabling Kalman filtering")

    # Save detailed comparison
    comparison_path = MODELS_V2 / "v1_v2_comparison.csv"
    improvements.to_csv(comparison_path, index=False)
    print(f"\n[SAVED] Detailed comparison: {comparison_path}")


if __name__ == "__main__":
    main()
