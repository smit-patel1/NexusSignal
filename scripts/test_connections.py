"""
Connection test script for NexusSignal.
Verifies PostgreSQL and Valkey connectivity using actual Phase 0 connectors.
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.config import settings
from backend.app.db import get_engine
from backend.app.cache import get_redis


async def test_postgresql() -> bool:
    """
    Test PostgreSQL connection using async SQLAlchemy engine.

    Returns:
        True if connection successful, False otherwise
    """
    print("\n" + "=" * 60)
    print("Testing PostgreSQL Connection")
    print("=" * 60)
    print(f"Database URL: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else settings.DATABASE_URL}")

    try:
        engine = get_engine()

        # Attempt to open and close a connection
        async with engine.connect() as conn:
            # Execute a simple query to verify connection
            from sqlalchemy import text
            result = await conn.execute(text("SELECT 1"))
            result.close()

        print("[OK] PostgreSQL connection OK")
        print("  - Connection opened successfully")
        print("  - Query executed successfully")
        print("  - Connection closed successfully")

        # Clean up
        await engine.dispose()

        return True

    except Exception as e:
        print("[FAIL] PostgreSQL connection FAILED")
        print(f"  Error: {type(e).__name__}")
        print(f"  Message: {str(e)}")
        return False


async def test_valkey() -> bool:
    """
    Test Valkey/Redis connection using async client.

    Returns:
        True if connection successful, False otherwise
    """
    print("\n" + "=" * 60)
    print("Testing Valkey/Redis Connection")
    print("=" * 60)

    # Parse URL for display (hide password)
    display_url = settings.REDIS_URL
    if "@" in display_url:
        parts = display_url.split("@")
        display_url = f"{parts[0].split(':')[0]}://***@{parts[1]}"

    print(f"Redis URL: {display_url}")

    try:
        redis_client = get_redis()

        # Test 1: Ping
        ping_result = await redis_client.ping()
        if not ping_result:
            raise Exception("Ping returned False")

        # Test 2: Set a test key
        test_key = "nexussignal:test:connection"
        test_value = "connection_test_ok"
        await redis_client.set(test_key, test_value, ex=60)

        # Test 3: Get the test key
        retrieved_value = await redis_client.get(test_key)
        if retrieved_value != test_value:
            raise Exception(f"Value mismatch: expected '{test_value}', got '{retrieved_value}'")

        # Test 4: Delete the test key
        await redis_client.delete(test_key)

        print("[OK] Valkey/Redis connection OK")
        print("  - Ping successful")
        print("  - Set operation successful")
        print("  - Get operation successful")
        print("  - Delete operation successful")

        # Clean up
        await redis_client.aclose()

        return True

    except Exception as e:
        print("[FAIL] Valkey/Redis connection FAILED")
        print(f"  Error: {type(e).__name__}")
        print(f"  Message: {str(e)}")
        return False


async def main() -> None:
    """Main entry point for connection tests."""
    # Load environment variables
    load_dotenv()

    print("\n" + "=" * 60)
    print("NexusSignal Connection Test")
    print("=" * 60)
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Log Level: {settings.LOG_LEVEL}")

    # Run tests sequentially
    postgres_ok = await test_postgresql()
    valkey_ok = await test_valkey()

    # Summary
    print("\n" + "=" * 60)
    print("Connection Test Summary")
    print("=" * 60)
    print(f"PostgreSQL: {'[PASS]' if postgres_ok else '[FAIL]'}")
    print(f"Valkey/Redis: {'[PASS]' if valkey_ok else '[FAIL]'}")

    if postgres_ok and valkey_ok:
        print("\n[SUCCESS] All connections successful!")
        print("You can now proceed with data ingestion.")
        sys.exit(0)
    else:
        print("\n[ERROR] One or more connections failed.")
        print("Please check your .env configuration and service availability.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
