"""
Cache connection management for NexusSignal.
Provides async Redis/Valkey client for caching operations.
"""

from redis.asyncio import Redis, from_url

from .config import settings


# Global Redis client instance
_redis_client: Redis | None = None


def get_redis() -> Redis:
    """Get or create the async Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


async def init_cache() -> None:
    """Initialize cache connection. Called on app startup."""
    client = get_redis()
    # Test connection
    await client.ping()


async def close_cache() -> None:
    """Close cache connection. Called on app shutdown."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


async def ping() -> bool:
    """
    Ping the cache server to verify connectivity.

    Returns:
        True if connection is healthy, False otherwise.
    """
    try:
        client = get_redis()
        result = await client.ping()
        return result
    except Exception:
        return False
