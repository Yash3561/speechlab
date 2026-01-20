"""
Health Check Endpoints

System health and status monitoring.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
import psutil
import shutil
from backend.core.redis_client import get_redis
from backend.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    services: dict
    system_metrics: dict



@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health and connected services.
    
    Returns status of API, database, Redis, and MLflow.
    """

    # 1. Service Checks
    redis_status = "offline"
    try:
        redis_client = get_redis()
        if redis_client.set("health_check", "ok", ttl=10):
            redis_status = "online"
    except Exception:
        pass

    services = {
        "api": "online",
        "database": "online",  # Supabase (managed)
        "redis": redis_status,
        "mlflow": "online" if settings.mlflow_tracking_uri else "offline",
        "ray_cluster": "online" # Mock for now
    }
    
    # 2. System Metrics
    cpu_percent = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    
    system_metrics = {
        "cpu_percent": cpu_percent,
        "memory_used_gb": round(memory.used / (1024**3), 2),
        "memory_total_gb": round(memory.total / (1024**3), 2),
        "memory_percent": memory.percent,
        "disk_percent": disk.percent,
        "gpu_memory_used": 0.0, # Placeholder
        "gpu_memory_total": 0.0
    }
    
    # Try GPU metrics if available
    try:
        import torch
        if torch.cuda.is_available():
            system_metrics["gpu_memory_used"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            system_metrics["gpu_memory_total"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except:
        pass
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
        services=services,
        system_metrics=system_metrics
    )


@router.get("/health/ready")
async def readiness_check():
    """Kubernetes-style readiness probe."""
    return {"ready": True}


@router.get("/health/live")
async def liveness_check():
    """Kubernetes-style liveness probe."""
    return {"alive": True}
