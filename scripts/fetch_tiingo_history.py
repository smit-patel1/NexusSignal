"""
Historical OHLCV data fetcher from Tiingo API.
Fetches 10+ years of daily bars and 3-5 years of 1H bars for configured tickers.
Saves data to Parquet files with proper deduplication.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import httpx
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.config import settings, TICKERS, TIMEFRAMES
from scripts.utils.parquet_utils import append_to_parquet, get_latest_timestamp


# Data paths
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "prices"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Tiingo API configuration
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo"
TIINGO_IEX_BASE_URL = "https://api.tiingo.com/iex"


class TiingoHistoricalFetcher:
    """Fetches historical price data from Tiingo API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {api_key}",
        }

    async def fetch_daily_history(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data for a ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data (default: 10 years ago)
            end_date: End date for data (default: today)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 10)
        if end_date is None:
            end_date = datetime.now()

        url = f"{TIINGO_BASE_URL}/daily/{ticker}/prices"
        params = {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "format": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()

            if not data:
                print(f"No data returned for {ticker} daily")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Rename columns to standard format
            df = df.rename(
                columns={
                    "date": "timestamp",
                    "adjOpen": "open",
                    "adjHigh": "high",
                    "adjLow": "low",
                    "adjClose": "close",
                    "adjVolume": "volume",
                }
            )

            # Select and order columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            print(f"Fetched {len(df)} daily bars for {ticker}")
            return df

        except httpx.HTTPStatusError as e:
            print(f"HTTP error fetching {ticker} daily: {e.response.status_code}")
            return None
        except Exception as e:
            print(f"Error fetching {ticker} daily: {e}")
            return None

    async def fetch_intraday_history(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resample_freq: str = "1hour",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday (1H) OHLCV data for a ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data (default: 3 years ago)
            end_date: End date for data (default: today)
            resample_freq: Resampling frequency (default: 1hour)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 3)
        if end_date is None:
            end_date = datetime.now()

        url = f"{TIINGO_IEX_BASE_URL}/{ticker}/prices"
        params = {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "resampleFreq": resample_freq,
            "format": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()

            if not data:
                print(f"No data returned for {ticker} intraday")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Rename columns to standard format
            df = df.rename(columns={"date": "timestamp"})

            # Select and order columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            print(f"Fetched {len(df)} 1H bars for {ticker}")
            return df

        except httpx.HTTPStatusError as e:
            print(f"HTTP error fetching {ticker} intraday: {e.response.status_code}")
            return None
        except Exception as e:
            print(f"Error fetching {ticker} intraday: {e}")
            return None

    async def fetch_and_save(self, ticker: str, timeframe: str) -> None:
        """
        Fetch historical data and save to Parquet file.

        Args:
            ticker: Stock ticker symbol
            timeframe: Either "daily" or "1h"
        """
        print(f"\n{'=' * 60}")
        print(f"Processing {ticker} - {timeframe}")
        print(f"{'=' * 60}")

        # Determine file path
        file_path = DATA_DIR / f"{ticker}_{timeframe}.parquet"

        # Check if file exists and get latest timestamp
        latest_timestamp = get_latest_timestamp(file_path)
        if latest_timestamp:
            print(f"Existing data found. Latest timestamp: {latest_timestamp}")
            start_date = latest_timestamp + timedelta(days=1)
        else:
            print("No existing data. Fetching full history.")
            start_date = None

        # Fetch data based on timeframe
        if timeframe == "daily":
            df = await self.fetch_daily_history(ticker, start_date=start_date)
        elif timeframe == "1h":
            df = await self.fetch_intraday_history(ticker, start_date=start_date)
        else:
            print(f"Unknown timeframe: {timeframe}")
            return

        if df is None or df.empty:
            print(f"No new data to save for {ticker} - {timeframe}")
            return

        # Add ticker column
        df.insert(0, "ticker", ticker)

        # Save/append to Parquet
        final_df = append_to_parquet(df, file_path, timestamp_col="timestamp")
        print(f"Saved {len(df)} new rows. Total rows: {len(final_df)}")
        print(f"File: {file_path}")

        # Rate limiting - be nice to the API
        await asyncio.sleep(0.5)


async def main() -> None:
    """Main entry point for historical data fetching."""
    # Load environment variables
    load_dotenv()

    # Check API key
    if not settings.TIINGO_API_KEY:
        print("ERROR: TIINGO_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return

    print(f"Starting historical data fetch for {len(TICKERS)} tickers")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Data directory: {DATA_DIR}")

    fetcher = TiingoHistoricalFetcher(settings.TIINGO_API_KEY)

    # Fetch data for each ticker and timeframe
    for ticker in TICKERS:
        for timeframe in TIMEFRAMES:
            try:
                await fetcher.fetch_and_save(ticker, timeframe)
            except Exception as e:
                print(f"Error processing {ticker} - {timeframe}: {e}")
                continue

    print("\n" + "=" * 60)
    print("Historical data fetch completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
