"""
Health check endpoint for NexusSignal API.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Status message indicating service health.
    """
    return {"status": "ok"}
