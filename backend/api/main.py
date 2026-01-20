"""
SpeechLab FastAPI Application

Main API entry point with WebSocket support for real-time metrics.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.core.logging import logger
from backend.api.endpoints import experiments, health, models, evaluation, training


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 Starting SpeechLab API...")
    logger.info(f"   MLflow URI: {settings.mlflow_tracking_uri}")
    logger.info(f"   Database: {settings.database_url.split('@')[-1]}")
    
    # Initialize Ray if configured
    if settings.ray_address:
        try:
            import ray
            ray.init(address=settings.ray_address, ignore_reinit_error=True)
            logger.info(f"   Ray: Connected to {settings.ray_address}")
        except Exception as e:
            logger.warning(f"   Ray: Could not connect - {e}")
    
    yield
    
    logger.info("👋 Shutting down SpeechLab API...")


# Create FastAPI app
app = FastAPI(
    title="SpeechLab API",
    description="Production-Grade Speech Model Training Infrastructure",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SpeechLab API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
