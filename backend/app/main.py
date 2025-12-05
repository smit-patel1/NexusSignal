"""
NexusSignal FastAPI Application.
Main entry point for the financial forecasting and signal system.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from .config import settings
from .db import init_db, close_db
from .cache import init_cache, close_cache
from .routers import health


# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting NexusSignal application...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    try:
        await init_db()
        logger.info("Database connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    try:
        await init_cache()
        logger.info("Cache connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")

    logger.info("NexusSignal startup complete")

    yield

    # Shutdown
    logger.info("Shutting down NexusSignal application...")

    try:
        await close_cache()
        logger.info("Cache connection closed")
    except Exception as e:
        logger.error(f"Error closing cache: {e}")

    try:
        await close_db()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")

    logger.info("NexusSignal shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="NexusSignal",
    description="Multi-engine financial forecasting and signal system",
    version="0.1.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(health.router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "NexusSignal API",
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
    }
