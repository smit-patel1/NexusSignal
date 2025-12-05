"""
Historical OHLCV data fetcher using Yahoo Finance.
Fetches 20 years of daily data and 2 years of hourly data for configured tickers.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.config import TICKERS
from scripts.utils.parquet_utils import write_parquet, read_parquet


# Data paths
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "prices"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class YahooHistoricalFetcher:
    """Fetches historical price data from Yahoo Finance."""

    def __init__(self):
        pass

    def fetch_daily_history(
        self,
        ticker: str,
        years: int = 20
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data for a ticker.

        Args:
            ticker: Stock ticker symbol
            years: Number of years of history (default: 20)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)

            print(f"  Fetching daily data from {start_date.date()} to {end_date.date()}")

            # Download data from Yahoo Finance
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                actions=False
            )

            if df.empty:
                print(f"  No data returned for {ticker} daily")
                return None

            # Reset index to get timestamp as column
            df = df.reset_index()

            # Rename columns to standard format
            df = df.rename(columns={
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Select only needed columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)

            print(f"  Fetched {len(df)} daily bars for {ticker}")
            return df

        except Exception as e:
            print(f"  Error fetching {ticker} daily: {type(e).__name__} - {e}")
            return None

    def fetch_hourly_history(
        self,
        ticker: str,
        years: int = 2
    ) -> Optional[pd.DataFrame]:
        """
        Fetch hourly OHLCV data for a ticker.

        Args:
            ticker: Stock ticker symbol
            years: Number of years of history (default: 2)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Yahoo Finance hourly data is limited to ~2 years
            # Fetch in chunks if needed
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)

            print(f"  Fetching hourly data from {start_date.date()} to {end_date.date()}")

            # Download data from Yahoo Finance
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date,
                end=end_date,
                interval="1h",
                auto_adjust=True,
                actions=False
            )

            if df.empty:
                print(f"  No data returned for {ticker} hourly")
                return None

            # Reset index to get timestamp as column
            df = df.reset_index()

            # Rename columns to standard format
            df = df.rename(columns={
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Select only needed columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            # Convert timestamp to datetime (remove timezone)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)

            print(f"  Fetched {len(df)} hourly bars for {ticker}")
            return df

        except Exception as e:
            print(f"  Error fetching {ticker} hourly: {type(e).__name__} - {e}")
            return None

    def fetch_and_save(self, ticker: str) -> tuple[bool, bool]:
        """
        Fetch historical data and save to Parquet files.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (daily_success, hourly_success)
        """
        print(f"\n{'=' * 60}")
        print(f"Processing {ticker}")
        print(f"{'=' * 60}")

        daily_success = False
        hourly_success = False

        # Fetch daily data
        daily_path = DATA_DIR / f"{ticker}_daily.parquet"
        try:
            df_daily = self.fetch_daily_history(ticker, years=20)

            if df_daily is not None and not df_daily.empty:
                # Add ticker column
                df_daily.insert(0, "ticker", ticker)

                # Save to Parquet
                write_parquet(df_daily, daily_path)
                print(f"  [OK] Saved daily data: {len(df_daily)} rows → {daily_path}")
                daily_success = True
            else:
                print(f"  [SKIP] No daily data to save")

        except Exception as e:
            print(f"  [ERROR] Failed to save daily data: {e}")

        # Fetch hourly data
        hourly_path = DATA_DIR / f"{ticker}_1h.parquet"
        try:
            df_hourly = self.fetch_hourly_history(ticker, years=2)

            if df_hourly is not None and not df_hourly.empty:
                # Add ticker column
                df_hourly.insert(0, "ticker", ticker)

                # Save to Parquet
                write_parquet(df_hourly, hourly_path)
                print(f"  [OK] Saved hourly data: {len(df_hourly)} rows → {hourly_path}")
                hourly_success = True
            else:
                print(f"  [SKIP] No hourly data to save")

        except Exception as e:
            print(f"  [ERROR] Failed to save hourly data: {e}")

        return daily_success, hourly_success


async def main() -> None:
    """Main entry point for Yahoo Finance historical data fetching."""
    # Load environment variables
    load_dotenv()

    print("\n" + "=" * 60)
    print("Yahoo Finance Historical Data Fetcher")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print(f"Tickers: {len(TICKERS)}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Timeframes: daily (20 years), hourly (2 years)")
    print("=" * 60)

    fetcher = YahooHistoricalFetcher()

    # Track results
    successful_tickers = []
    failed_tickers = []
    daily_count = 0
    hourly_count = 0

    # Fetch data for each ticker
    for i, ticker in enumerate(TICKERS, 1):
        print(f"\n[{i}/{len(TICKERS)}] {ticker}")

        try:
            daily_ok, hourly_ok = fetcher.fetch_and_save(ticker)

            if daily_ok:
                daily_count += 1
            if hourly_ok:
                hourly_count += 1

            if daily_ok and hourly_ok:
                successful_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)

        except Exception as e:
            print(f"  [ERROR] Unexpected error: {e}")
            failed_tickers.append(ticker)

    # Final summary
    print("\n" + "=" * 60)
    print("Historical Data Fetch Complete")
    print("=" * 60)
    print(f"Successful tickers: {len(successful_tickers)}/{len(TICKERS)}")
    print(f"Daily files created: {daily_count}")
    print(f"Hourly files created: {hourly_count}")

    if successful_tickers:
        print(f"\n[SUCCESS] {', '.join(successful_tickers[:10])}"
              f"{'...' if len(successful_tickers) > 10 else ''}")

    if failed_tickers:
        print(f"\n[FAILED] {', '.join(failed_tickers[:10])}"
              f"{'...' if len(failed_tickers) > 10 else ''}")

    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
