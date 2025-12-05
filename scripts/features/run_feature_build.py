"""
Automated feature building runner for NexusSignal.
Processes all configured tickers and generates feature datasets.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.app.config import TICKERS
from scripts.features.build_features import build_features_for_ticker, get_feature_summary


def main() -> None:
    """Main entry point for automated feature building."""
    print("\n" + "=" * 60)
    print("NexusSignal Feature Builder")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print(f"Tickers to process: {len(TICKERS)}")
    print(f"Tickers: {', '.join(TICKERS[:5])}{'...' if len(TICKERS) > 5 else ''}")
    print("=" * 60)

    # Track results
    successful = []
    failed = []

    # Process each ticker
    for i, ticker in enumerate(TICKERS, 1):
        print(f"\n[{i}/{len(TICKERS)}] Processing {ticker}...")

        try:
            # Build features
            df = build_features_for_ticker(ticker)

            # Get summary
            summary = get_feature_summary(ticker)

            successful.append(ticker)

            print(f"[OK] {ticker}: {summary['total_rows']} rows, {summary['total_columns']} features")

        except FileNotFoundError as e:
            print(f"[SKIP] {ticker}: {e}")
            failed.append((ticker, "Missing raw data"))

        except Exception as e:
            print(f"[ERROR] {ticker}: {type(e).__name__} - {e}")
            failed.append((ticker, str(e)))

    # Final summary
    print("\n" + "=" * 60)
    print("Feature Building Complete")
    print("=" * 60)
    print(f"Successful: {len(successful)}/{len(TICKERS)}")
    print(f"Failed: {len(failed)}/{len(TICKERS)}")

    if successful:
        print(f"\n[SUCCESS] Processed tickers:")
        for ticker in successful:
            summary = get_feature_summary(ticker)
            print(f"  - {ticker}: {summary['total_rows']} rows, {summary['total_columns']} columns")

    if failed:
        print(f"\n[FAILED] Tickers:")
        for ticker, reason in failed:
            print(f"  - {ticker}: {reason}")

    print("\n" + "=" * 60)
    print("Feature datasets saved to: data/processed/features/")
    print("=" * 60)


if __name__ == "__main__":
    main()
