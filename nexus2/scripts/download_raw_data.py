#!/usr/bin/env python
"""
Download historical OHLCV price data for NexusSignal tickers using yfinance.

This script downloads 1-hour interval data for the past 5 years and saves
each ticker as a parquet file in the raw data directory.

Usage:
    python nexus2/scripts/download_raw_data.py
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


# Hardcoded list of NexusSignal tickers
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "V",
    "UNH", "JNJ", "JPM", "LLY", "XOM", "PG", "HD", "ABBV",
    "CVX", "MRK", "PEP", "KO", "AVGO", "COST"
]

# Download parameters
# Note: Yahoo Finance limits 1h data to the last 730 days
PERIOD = "730d"
INTERVAL = "1h"

# Column mapping from yfinance default names to required names
COLUMN_MAPPING = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
}


def get_repo_root() -> Path:
    """Get the repository root directory."""
    # Script is at nexus2/scripts/download_raw_data.py
    # Repo root is two levels up
    return Path(__file__).resolve().parent.parent.parent


def download_ticker_data(ticker: str) -> pd.DataFrame | None:
    """
    Download OHLCV data for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with OHLCV data or None if download failed/empty
    """
    print(f"Downloading {ticker}...")
    
    try:
        # Download data using yfinance
        data = yf.download(
            ticker,
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
            auto_adjust=True  # Use adjusted prices
        )
        
        # Check if download returned empty dataframe
        if data.empty:
            print(f"  WARNING: Empty dataframe returned for {ticker}")
            return None
        
        # Handle MultiIndex columns (yfinance returns MultiIndex when downloading single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex - take just the first level (Open, High, Low, etc.)
            data.columns = data.columns.get_level_values(0)
        
        # Select and rename only the OHLCV columns we need
        columns_to_keep = list(COLUMN_MAPPING.keys())
        available_columns = [col for col in columns_to_keep if col in data.columns]
        
        if len(available_columns) != len(columns_to_keep):
            missing = set(columns_to_keep) - set(available_columns)
            print(f"  WARNING: Missing columns for {ticker}: {missing}")
            return None
        
        # Select only OHLCV columns
        data = data[columns_to_keep].copy()
        
        # Rename columns to lowercase
        data.rename(columns=COLUMN_MAPPING, inplace=True)
        
        # Ensure index is a DatetimeIndex and named 'timestamp'
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data.index.name = "timestamp"
        
        # Sort by timestamp (ascending)
        data.sort_index(inplace=True)
        
        # Drop any rows with NaN values in OHLCV columns
        data.dropna(inplace=True)
        
        if data.empty:
            print(f"  WARNING: No valid data after cleaning for {ticker}")
            return None
        
        print(f"  Downloaded {len(data)} rows for {ticker}")
        return data
        
    except Exception as e:
        print(f"  ERROR downloading {ticker}: {e}")
        return None


def save_ticker_data(data: pd.DataFrame, ticker: str, output_dir: Path) -> bool:
    """
    Save ticker data to parquet file.
    
    Args:
        data: DataFrame with OHLCV data
        ticker: Stock ticker symbol
        output_dir: Directory to save the parquet file
        
    Returns:
        True if save successful, False otherwise
    """
    try:
        output_path = output_dir / f"{ticker}_1h.parquet"
        data.to_parquet(output_path, index=True)
        print(f"  Saved to {output_path}")
        return True
    except Exception as e:
        print(f"  ERROR saving {ticker}: {e}")
        return False


def main():
    """Main function to download and save all ticker data."""
    print("=" * 60)
    print("NexusSignal Raw Data Downloader")
    print("=" * 60)
    print(f"Tickers: {len(TICKERS)}")
    print(f"Period: {PERIOD}")
    print(f"Interval: {INTERVAL}")
    print("=" * 60)
    
    # Setup output directory
    repo_root = get_repo_root()
    output_dir = repo_root / "data" / "raw" / "prices"
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Track results
    successful = []
    failed = []
    
    # Download and save each ticker
    for ticker in TICKERS:
        data = download_ticker_data(ticker)
        
        if data is not None:
            if save_ticker_data(data, ticker, output_dir):
                successful.append(ticker)
            else:
                failed.append(ticker)
        else:
            failed.append(ticker)
        
        print()  # Add blank line between tickers
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successfully saved: {len(successful)}/{len(TICKERS)} files")
    
    if successful:
        print(f"  Saved: {', '.join(successful)}")
    
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    
    print("=" * 60)
    
    # Return exit code based on success
    if len(failed) == len(TICKERS):
        print("ERROR: All downloads failed!")
        return 1
    elif failed:
        print("WARNING: Some downloads failed")
        return 0  # Partial success is still success
    else:
        print("SUCCESS: All downloads completed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

