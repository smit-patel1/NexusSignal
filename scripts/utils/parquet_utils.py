"""
Parquet utility functions for reading, writing, and managing price data.
Handles deduplication, appending, and maintaining sorted indices.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_parquet(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Read a Parquet file into a DataFrame.

    Args:
        file_path: Path to the Parquet file

    Returns:
        DataFrame if file exists, None otherwise
    """
    if not file_path.exists():
        return None

    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def write_parquet(df: pd.DataFrame, file_path: Path) -> None:
    """
    Write a DataFrame to a Parquet file.

    Args:
        df: DataFrame to write
        file_path: Path to save the Parquet file
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with compression
    df.to_parquet(
        file_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )


def append_to_parquet(
    new_data: pd.DataFrame,
    file_path: Path,
    timestamp_col: str = "timestamp",
    deduplicate: bool = True,
) -> pd.DataFrame:
    """
    Append new data to an existing Parquet file, handling deduplication.

    Args:
        new_data: New DataFrame to append
        file_path: Path to the existing Parquet file
        timestamp_col: Name of the timestamp column for sorting and deduplication
        deduplicate: Whether to remove duplicate timestamps

    Returns:
        Combined DataFrame with all data
    """
    # Read existing data if file exists
    existing_data = read_parquet(file_path)

    if existing_data is None or existing_data.empty:
        combined = new_data.copy()
    else:
        # Combine existing and new data
        combined = pd.concat([existing_data, new_data], ignore_index=True)

    # Ensure timestamp column is datetime
    if timestamp_col in combined.columns:
        combined[timestamp_col] = pd.to_datetime(combined[timestamp_col])

        # Sort by timestamp
        combined = combined.sort_values(timestamp_col).reset_index(drop=True)

        # Remove duplicates if requested
        if deduplicate:
            combined = combined.drop_duplicates(
                subset=[timestamp_col],
                keep="last"
            ).reset_index(drop=True)

    # Write back to file
    write_parquet(combined, file_path)

    return combined


def get_latest_timestamp(
    file_path: Path,
    timestamp_col: str = "timestamp"
) -> Optional[pd.Timestamp]:
    """
    Get the latest timestamp from a Parquet file.

    Args:
        file_path: Path to the Parquet file
        timestamp_col: Name of the timestamp column

    Returns:
        Latest timestamp if file exists and has data, None otherwise
    """
    df = read_parquet(file_path)

    if df is None or df.empty:
        return None

    if timestamp_col not in df.columns:
        return None

    # Ensure timestamp column is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    return df[timestamp_col].max()


def deduplicate_parquet(
    file_path: Path,
    timestamp_col: str = "timestamp"
) -> None:
    """
    Remove duplicate timestamps from a Parquet file in-place.

    Args:
        file_path: Path to the Parquet file
        timestamp_col: Name of the timestamp column
    """
    df = read_parquet(file_path)

    if df is None or df.empty:
        return

    if timestamp_col not in df.columns:
        return

    # Ensure timestamp column is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Sort and remove duplicates
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df = df.drop_duplicates(subset=[timestamp_col], keep="last").reset_index(drop=True)

    # Write back
    write_parquet(df, file_path)


def get_row_count(file_path: Path) -> int:
    """
    Get the number of rows in a Parquet file without loading into memory.

    Args:
        file_path: Path to the Parquet file

    Returns:
        Number of rows, or 0 if file doesn't exist
    """
    if not file_path.exists():
        return 0

    try:
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows
    except Exception:
        return 0
