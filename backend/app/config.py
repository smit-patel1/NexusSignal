"""
Configuration management for NexusSignal.
Loads environment variables and provides typed configuration access.
"""

from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://localhost/nexussignal",
        description="PostgreSQL database URL with asyncpg driver"
    )

    # Cache
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis/Valkey connection URL"
    )

    # Environment
    ENVIRONMENT: Literal["dev", "prod"] = Field(
        default="dev",
        description="Application environment"
    )

    LOG_LEVEL: str = Field(
        default="info",
        description="Logging level"
    )

    # External API Keys
    TIINGO_API_KEY: str = Field(
        default="",
        description="Tiingo API key for financial data"
    )

    FINNHUB_API_KEY: str = Field(
        default="",
        description="Finnhub API key for market data"
    )

    NEWSAPI_API_KEY: str = Field(
        default="",
        description="NewsAPI key for news data"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Trading configuration
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "V", "XOM", "WMT", "JPM", "MA",
    "PG", "CVX", "LLY", "HD", "MRK",
    "ABBV", "KO", "PEP", "AVGO", "COST"
]

TIMEFRAMES = ["daily", "1h"]
