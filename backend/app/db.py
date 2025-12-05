"""
Database connection management for NexusSignal.
Provides async SQLAlchemy engine and session management.
"""

from datetime import datetime
from typing import AsyncGenerator
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base

from .config import settings


# Base class for SQLAlchemy models
Base = declarative_base()


# Database models
class PriceLive(Base):
    """Live price data table for latest market data."""
    __tablename__ = "prices_live"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        Index("idx_ticker_timestamp", "ticker", "timestamp"),
    )


class NewsLive(Base):
    """Live news table for recent headlines and sentiment."""
    __tablename__ = "news_live"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    headline = Column(Text, nullable=False)
    sentiment_score = Column(Float, nullable=True)
    source = Column(String(100), nullable=False)
    url = Column(Text, nullable=True, unique=True)

    __table_args__ = (
        Index("idx_ticker_timestamp_news", "ticker", "timestamp"),
    )


# Global engine instance
_engine: AsyncEngine | None = None
_async_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async database engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.ENVIRONMENT == "dev",
            future=True,
            pool_pre_ping=True,
        )
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session maker."""
    global _async_session_maker
    if _async_session_maker is None:
        _async_session_maker = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_maker


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get database sessions.

    Usage:
        @app.get("/example")
        async def example(db: AsyncSession = Depends(get_session)):
            ...
    """
    async_session = get_session_maker()
    async with async_session() as session:
        yield session


async def init_db() -> None:
    """Initialize database connection. Called on app startup."""
    engine = get_engine()
    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connection. Called on app shutdown."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
