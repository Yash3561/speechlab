"""
Health Check Endpoints

System health and status monitoring.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    services: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health and connected services.
    
    Returns status of API, database, Redis, and MLflow.
    """
    services = {
        "api": "ok",
        "database": "ok",  # TODO: Add actual check
        "redis": "ok",      # TODO: Add actual check
        "mlflow": "ok",     # TODO: Add actual check
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
        services=services,
    )


@router.get("/health/ready")
async def readiness_check():
    """Kubernetes-style readiness probe."""
    return {"ready": True}


@router.get("/health/live")
async def liveness_check():
    """Kubernetes-style liveness probe."""
    return {"alive": True}
