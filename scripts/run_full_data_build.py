"""
Automated end-to-end data preparation pipeline for NexusSignal.
Runs Phase 1 (data ingestion) and Phase 2 (feature engineering) with validation.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.config import settings, TICKERS
from backend.app.db import get_engine, PriceLive, NewsLive
from backend.app.cache import get_redis
from sqlalchemy import select, func


# Import Phase 1 scripts
import scripts.fetch_history_yahoo as phase1_history
import scripts.update_tiingo_intraday as phase1_intraday
import scripts.fetch_news as phase1_news

# Import Phase 2 script
import scripts.features.run_feature_build as phase2_features

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_PRICES_DIR = DATA_DIR / "raw" / "prices"
PROCESSED_FEATURES_DIR = DATA_DIR / "processed" / "features"


async def test_postgres() -> bool:
    """Test PostgreSQL connectivity and create tables."""
    print("\n" + "=" * 60)
    print("[TEST] PostgreSQL Connection")
    print("=" * 60)

    try:
        from backend.app.db import init_db

        # Initialize database (creates tables)
        await init_db()

        print("[OK] PostgreSQL connection successful")
        print("[OK] Database tables created/verified")
        return True

    except Exception as e:
        print(f"[FAIL] PostgreSQL connection failed: {e}")
        return False


async def test_valkey() -> bool:
    """Test Valkey/Redis connectivity."""
    print("\n" + "=" * 60)
    print("[TEST] Valkey/Redis Connection")
    print("=" * 60)

    try:
        redis_client = get_redis()
        result = await asyncio.wait_for(redis_client.ping(), timeout=5.0)

        if not result:
            raise Exception("Ping returned False")

        await redis_client.aclose()
        print("[OK] Valkey/Redis connection successful")
        return True

    except asyncio.TimeoutError:
        print(f"[FAIL] Valkey/Redis connection timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Valkey/Redis connection failed: {e}")
        return False


async def run_phase1_history() -> None:
    """Run Phase 1: Historical OHLCV data fetch (Yahoo Finance)."""
    print("\n" + "=" * 60)
    print("[Phase 1] Fetching Historical OHLCV Data (Yahoo Finance)")
    print("=" * 60)

    try:
        await phase1_history.main()
        print("[OK] Historical data fetch completed")
    except Exception as e:
        print(f"[ERROR] Historical data fetch failed: {e}")
        raise


async def run_phase1_intraday() -> None:
    """Run Phase 1: Intraday data update."""
    print("\n" + "=" * 60)
    print("[Phase 1] Updating Intraday Candles")
    print("=" * 60)

    try:
        await phase1_intraday.main()
        print("[OK] Intraday update completed")
    except Exception as e:
        print(f"[ERROR] Intraday update failed: {e}")
        raise


async def run_phase1_news() -> None:
    """Run Phase 1: News data fetch."""
    print("\n" + "=" * 60)
    print("[Phase 1] Fetching News Data")
    print("=" * 60)

    try:
        await phase1_news.main()
        print("[OK] News fetch completed")
    except Exception as e:
        print(f"[ERROR] News fetch failed: {e}")
        raise


def run_phase2_features() -> None:
    """Run Phase 2: Feature engineering."""
    print("\n" + "=" * 60)
    print("[Phase 2] Building Features")
    print("=" * 60)

    try:
        phase2_features.main()
        print("[OK] Feature building completed")
    except Exception as e:
        print(f"[ERROR] Feature building failed: {e}")
        raise


def validate_raw_data() -> Tuple[List[str], List[str]]:
    """
    Validate raw Parquet files exist.

    Returns:
        Tuple of (successful_tickers, missing_tickers)
    """
    print("\n" + "=" * 60)
    print("[VALIDATION] Raw Parquet Files")
    print("=" * 60)

    successful = []
    missing = []

    for ticker in TICKERS:
        daily_path = RAW_PRICES_DIR / f"{ticker}_daily.parquet"
        hourly_path = RAW_PRICES_DIR / f"{ticker}_1h.parquet"

        if daily_path.exists() and hourly_path.exists():
            successful.append(ticker)
            print(f"[OK] {ticker}: daily + hourly data found")
        else:
            missing.append(ticker)
            missing_files = []
            if not daily_path.exists():
                missing_files.append("daily")
            if not hourly_path.exists():
                missing_files.append("hourly")
            print(f"[MISSING] {ticker}: {', '.join(missing_files)}")

    print(f"\nRaw Data Summary: {len(successful)}/{len(TICKERS)} tickers complete")
    return successful, missing


def validate_features() -> Tuple[List[str], List[str]]:
    """
    Validate processed feature files exist.

    Returns:
        Tuple of (successful_tickers, missing_tickers)
    """
    print("\n" + "=" * 60)
    print("[VALIDATION] Processed Feature Files")
    print("=" * 60)

    successful = []
    missing = []

    for ticker in TICKERS:
        feature_path = PROCESSED_FEATURES_DIR / f"{ticker}_1h.parquet"

        if feature_path.exists():
            successful.append(ticker)

            # Get file size
            size_mb = feature_path.stat().st_size / (1024 * 1024)
            print(f"[OK] {ticker}: {size_mb:.2f} MB")
        else:
            missing.append(ticker)
            print(f"[MISSING] {ticker}")

    print(f"\nFeature Files Summary: {len(successful)}/{len(TICKERS)} tickers complete")
    return successful, missing


async def validate_postgres() -> Dict[str, int]:
    """
    Validate PostgreSQL contains data.

    Returns:
        Dictionary with table row counts
    """
    print("\n" + "=" * 60)
    print("[VALIDATION] PostgreSQL Data")
    print("=" * 60)

    try:
        engine = get_engine()

        async with engine.connect() as conn:
            # Count prices_live rows
            prices_result = await conn.execute(
                select(func.count()).select_from(PriceLive)
            )
            prices_count = prices_result.scalar()

            # Count news_live rows
            news_result = await conn.execute(
                select(func.count()).select_from(NewsLive)
            )
            news_count = news_result.scalar()

        await engine.dispose()

        print(f"[OK] prices_live: {prices_count} rows")
        print(f"[OK] news_live: {news_count} rows")

        return {
            "prices_live": prices_count,
            "news_live": news_count,
        }

    except Exception as e:
        print(f"[ERROR] PostgreSQL validation failed: {e}")
        return {}


async def validate_valkey() -> Dict[str, bool]:
    """
    Validate Valkey contains cached data.

    Returns:
        Dictionary with cache key existence
    """
    print("\n" + "=" * 60)
    print("[VALIDATION] Valkey Cache")
    print("=" * 60)

    try:
        redis_client = get_redis()
        results = {}

        for ticker in TICKERS[:5]:  # Sample first 5 tickers
            key = f"prices:{ticker}:1h"
            exists = await redis_client.exists(key)
            results[ticker] = bool(exists)

            if exists:
                print(f"[OK] {ticker}: cache key exists")
            else:
                print(f"[MISSING] {ticker}: cache key not found")

        await redis_client.aclose()

        cached_count = sum(results.values())
        print(f"\nCache Summary: {cached_count}/{len(results)} sampled tickers cached")

        return results

    except Exception as e:
        print(f"[ERROR] Valkey validation failed: {e}")
        return {}


async def validate_all() -> Dict:
    """
    Run all validation checks.

    Returns:
        Dictionary with validation results
    """
    print("\n" + "=" * 60)
    print("[VALIDATION] Starting Validation")
    print("=" * 60)

    # Validate raw data
    raw_success, raw_missing = validate_raw_data()

    # Validate features
    feature_success, feature_missing = validate_features()

    # Validate PostgreSQL
    postgres_counts = await validate_postgres()

    # Validate Valkey
    valkey_results = await validate_valkey()

    return {
        "raw_data": {
            "successful": raw_success,
            "missing": raw_missing,
        },
        "features": {
            "successful": feature_success,
            "missing": feature_missing,
        },
        "postgres": postgres_counts,
        "valkey": valkey_results,
    }


async def main() -> None:
    """Main orchestrator for full data pipeline."""
    start_time = datetime.now()

    print("\n" + "=" * 80)
    print("NexusSignal - Full Data Pipeline")
    print("=" * 80)
    print(f"Start Time: {start_time}")
    print(f"Tickers: {len(TICKERS)}")
    print("=" * 80)

    # Step 0: Test connections
    print("\n[STEP 0] Testing Connections")
    postgres_ok = await test_postgres()
    valkey_ok = await test_valkey()

    if not postgres_ok:
        print("\n[ABORT] PostgreSQL connection failed. Cannot proceed.")
        sys.exit(1)

    if not valkey_ok:
        print("\n[WARNING] Valkey connection failed. Caching will not work.")

    # Step 1: Run Phase 1 - Data Ingestion
    print("\n[STEP 1] Running Phase 1 - Data Ingestion")

    try:
        await run_phase1_history()
    except Exception as e:
        print(f"\n[ERROR] Phase 1 (history) failed: {e}")
        print("Continuing with next step...")

    try:
        await run_phase1_intraday()
    except Exception as e:
        print(f"\n[ERROR] Phase 1 (intraday) failed: {e}")
        print("Continuing with next step...")

    try:
        await run_phase1_news()
    except Exception as e:
        print(f"\n[ERROR] Phase 1 (news) failed: {e}")
        print("Continuing with next step...")

    # Step 2: Run Phase 2 - Feature Engineering
    print("\n[STEP 2] Running Phase 2 - Feature Engineering")

    try:
        run_phase2_features()
    except Exception as e:
        print(f"\n[ERROR] Phase 2 failed: {e}")
        print("Continuing with validation...")

    # Step 3: Validate all outputs
    print("\n[STEP 3] Validation")
    validation_results = await validate_all()

    # Step 4: Final Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print()

    # Raw data summary
    raw_success = len(validation_results["raw_data"]["successful"])
    raw_total = len(TICKERS)
    print(f"Raw Data: {raw_success}/{raw_total} tickers ({raw_success/raw_total*100:.1f}%)")

    if validation_results["raw_data"]["missing"]:
        print(f"  Missing: {', '.join(validation_results['raw_data']['missing'][:5])}"
              f"{'...' if len(validation_results['raw_data']['missing']) > 5 else ''}")

    # Feature data summary
    feature_success = len(validation_results["features"]["successful"])
    print(f"Features: {feature_success}/{raw_total} tickers ({feature_success/raw_total*100:.1f}%)")

    if validation_results["features"]["missing"]:
        print(f"  Missing: {', '.join(validation_results['features']['missing'][:5])}"
              f"{'...' if len(validation_results['features']['missing']) > 5 else ''}")

    # Database summary
    if validation_results["postgres"]:
        print(f"PostgreSQL:")
        print(f"  prices_live: {validation_results['postgres'].get('prices_live', 0)} rows")
        print(f"  news_live: {validation_results['postgres'].get('news_live', 0)} rows")

    # Cache summary
    if validation_results["valkey"]:
        cached = sum(validation_results["valkey"].values())
        total_checked = len(validation_results["valkey"])
        print(f"Valkey Cache: {cached}/{total_checked} sampled keys exist")

    # Final status
    print()
    if raw_success == raw_total and feature_success == raw_total:
        print("[COMPLETE] ✓ Full data pipeline executed successfully!")
    elif raw_success > 0 or feature_success > 0:
        print("[PARTIAL] ⚠ Pipeline completed with some missing data.")
    else:
        print("[FAILED] ✗ Pipeline did not produce expected outputs.")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
