"""
Incremental intraday price updater for NexusSignal.
Runs periodically (e.g., every 30 minutes via cron) to:
- Fetch only new candles since last update
- Append to Parquet files
- Write latest bars to Postgres prices_live table
- Cache latest window in Valkey/Redis
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import httpx
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.config import settings, TICKERS
from backend.app.db import get_engine, get_session_maker, PriceLive
from backend.app.cache import get_redis
from scripts.utils.parquet_utils import append_to_parquet, get_latest_timestamp


# Data paths
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "prices"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Tiingo API configuration
TIINGO_IEX_BASE_URL = "https://api.tiingo.com/iex"

# Cache configuration
CACHE_WINDOW_HOURS = 24  # Store last 24 hours in cache


class TiingoIncrementalUpdater:
    """Manages incremental updates of intraday price data."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {api_key}",
        }

    async def fetch_recent_bars(
        self,
        ticker: str,
        start_date: datetime,
        resample_freq: str = "1hour",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch recent intraday bars since a given start date.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data fetch
            resample_freq: Resampling frequency (default: 1hour)

        Returns:
            DataFrame with recent bars or None
        """
        url = f"{TIINGO_IEX_BASE_URL}/{ticker}/prices"
        params = {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "resampleFreq": resample_freq,
            "format": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()

            if not data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df = df.rename(columns={"date": "timestamp"})
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            return df

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"Rate limit hit for {ticker}. Waiting...")
                await asyncio.sleep(5)
            else:
                print(f"HTTP error fetching {ticker}: {e.response.status_code}")
            return None
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None

    async def update_parquet(
        self,
        ticker: str,
        timeframe: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """
        Update Parquet file with new data since last timestamp.

        Args:
            ticker: Stock ticker symbol
            timeframe: Timeframe (default: 1h)

        Returns:
            DataFrame with new data, or None if no updates
        """
        file_path = DATA_DIR / f"{ticker}_{timeframe}.parquet"

        # Get latest timestamp from existing file
        latest_timestamp = get_latest_timestamp(file_path)

        if latest_timestamp:
            # Fetch data starting from latest timestamp + 1 hour
            start_date = latest_timestamp + timedelta(hours=1)
            print(f"{ticker}: Last timestamp {latest_timestamp}, fetching from {start_date}")
        else:
            # No existing data, fetch last 7 days
            start_date = datetime.now() - timedelta(days=7)
            print(f"{ticker}: No existing data, fetching last 7 days")

        # Fetch new data
        df = await self.fetch_recent_bars(ticker, start_date)

        if df is None or df.empty:
            print(f"{ticker}: No new data")
            return None

        # Add ticker column
        df.insert(0, "ticker", ticker)

        # Append to Parquet
        final_df = append_to_parquet(df, file_path, timestamp_col="timestamp")
        print(f"{ticker}: Added {len(df)} new bars. Total: {len(final_df)}")

        return df

    async def update_postgres(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Write latest bars to Postgres prices_live table.

        Args:
            df: DataFrame with new price data
            ticker: Stock ticker symbol
        """
        if df is None or df.empty:
            return

        # Get the latest bar
        latest_bar = df.sort_values("timestamp").iloc[-1]

        # Prepare record
        record = {
            "ticker": ticker,
            "timestamp": latest_bar["timestamp"],
            "open": float(latest_bar["open"]),
            "high": float(latest_bar["high"]),
            "low": float(latest_bar["low"]),
            "close": float(latest_bar["close"]),
            "volume": float(latest_bar["volume"]),
        }

        # Insert or update in database
        try:
            engine = get_engine()
            session_maker = get_session_maker()

            async with session_maker() as session:
                # Use upsert (insert or update on conflict)
                stmt = insert(PriceLive).values(**record)
                stmt = stmt.on_conflict_do_nothing()

                await session.execute(stmt)
                await session.commit()

            print(f"{ticker}: Updated Postgres with latest bar at {record['timestamp']}")

        except Exception as e:
            print(f"{ticker}: Error updating Postgres: {e}")

    async def update_cache(self, ticker: str, timeframe: str = "1h") -> None:
        """
        Cache latest window of bars in Redis/Valkey for fast dashboard access.

        Args:
            ticker: Stock ticker symbol
            timeframe: Timeframe (default: 1h)
        """
        file_path = DATA_DIR / f"{ticker}_{timeframe}.parquet"

        try:
            # Read latest data from Parquet
            from scripts.utils.parquet_utils import read_parquet
            df = read_parquet(file_path)

            if df is None or df.empty:
                return

            # Get last N hours of data
            cutoff = datetime.now() - timedelta(hours=CACHE_WINDOW_HOURS)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            recent_df = df[df["timestamp"] >= cutoff].copy()

            if recent_df.empty:
                return

            # Convert to JSON-serializable format
            recent_df["timestamp"] = recent_df["timestamp"].astype(str)
            data_json = recent_df.to_dict(orient="records")

            # Store in Redis
            redis_client = get_redis()
            cache_key = f"prices:{ticker}:{timeframe}"

            await redis_client.set(
                cache_key,
                json.dumps(data_json),
                ex=3600 * 25  # Expire after 25 hours
            )

            print(f"{ticker}: Cached {len(recent_df)} bars (last {CACHE_WINDOW_HOURS}h)")

        except Exception as e:
            print(f"{ticker}: Error updating cache: {e}")

    async def update_ticker(self, ticker: str) -> None:
        """
        Run full update pipeline for a single ticker.

        Args:
            ticker: Stock ticker symbol
        """
        print(f"\n{'=' * 50}")
        print(f"Updating {ticker}")
        print(f"{'=' * 50}")

        # Update Parquet file
        new_data = await self.update_parquet(ticker, timeframe="1h")

        # Update Postgres with latest bar
        if new_data is not None and not new_data.empty:
            await self.update_postgres(new_data, ticker)

        # Update cache
        await self.update_cache(ticker, timeframe="1h")

        # Rate limiting
        await asyncio.sleep(0.5)


async def main() -> None:
    """Main entry point for incremental updates."""
    # Load environment variables
    load_dotenv()

    # Check API key
    if not settings.TIINGO_API_KEY:
        print("ERROR: TIINGO_API_KEY not found in environment variables")
        return

    print(f"Starting incremental update for {len(TICKERS)} tickers")
    print(f"Timestamp: {datetime.now()}")
    print(f"Data directory: {DATA_DIR}")

    updater = TiingoIncrementalUpdater(settings.TIINGO_API_KEY)

    # Update each ticker
    for ticker in TICKERS:
        try:
            await updater.update_ticker(ticker)
        except Exception as e:
            print(f"Error updating {ticker}: {e}")
            continue

    print("\n" + "=" * 50)
    print("Incremental update completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
