"""
Experiments API Endpoints

CRUD operations for training experiments with real-time metrics streaming.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import json

from backend.core.logging import logger
from backend.core.utils import generate_run_id

router = APIRouter()


# ============================================================
# Pydantic Models
# ============================================================

class ExperimentConfig(BaseModel):
    """Experiment configuration."""
    name: str
    model_architecture: str = "whisper"
    model_variant: str = "tiny"
    batch_size: int = 8
    learning_rate: float = 0.0001
    max_epochs: int = 10
    mixed_precision: bool = True
    gradient_accumulation: int = 4
    tags: List[str] = []


class ExperimentStatus(BaseModel):
    """Current experiment status."""
    id: str
    name: str
    status: str  # pending, running, completed, failed
    created_at: str
    updated_at: str
    current_epoch: int = 0
    total_epochs: int = 10
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    wer: Optional[float] = None
    progress: float = 0.0


class MetricsUpdate(BaseModel):
    """Real-time metrics update."""
    epoch: int
    step: int
    train_loss: float
    val_loss: Optional[float] = None
    wer: Optional[float] = None
    learning_rate: float
    gpu_util: float = 0.0
    throughput: float = 0.0  # samples/sec
    timestamp: str


# ============================================================
# In-Memory Storage (Replace with DB in production)
# ============================================================

experiments_db: dict = {}
active_connections: List[WebSocket] = []


# ============================================================
# REST Endpoints
# ============================================================

@router.get("/", response_model=List[ExperimentStatus])
async def list_experiments():
    """List all experiments."""
    return list(experiments_db.values())


@router.post("/", response_model=ExperimentStatus)
async def create_experiment(config: ExperimentConfig):
    """
    Create a new experiment.
    
    This queues the experiment for training.
    """
    experiment_id = generate_run_id("exp")
    now = datetime.utcnow().isoformat()
    
    experiment = ExperimentStatus(
        id=experiment_id,
        name=config.name,
        status="pending",
        created_at=now,
        updated_at=now,
        total_epochs=config.max_epochs,
    )
    
    experiments_db[experiment_id] = experiment.model_dump()
    logger.info(f"Created experiment: {experiment_id}")
    
    return experiment


@router.get("/{experiment_id}", response_model=ExperimentStatus)
async def get_experiment(experiment_id: str):
    """Get experiment status by ID."""
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiments_db[experiment_id]


@router.post("/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start a pending experiment."""
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiments_db[experiment_id]["status"] = "running"
    experiments_db[experiment_id]["updated_at"] = datetime.utcnow().isoformat()
    
    # TODO: Queue training job with Ray
    logger.info(f"Started experiment: {experiment_id}")
    
    return {"message": "Experiment started", "id": experiment_id}


@router.post("/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop a running experiment."""
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiments_db[experiment_id]["status"] = "stopped"
    experiments_db[experiment_id]["updated_at"] = datetime.utcnow().isoformat()
    
    # TODO: Cancel Ray job
    logger.info(f"Stopped experiment: {experiment_id}")
    
    return {"message": "Experiment stopped", "id": experiment_id}


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment."""
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    del experiments_db[experiment_id]
    logger.info(f"Deleted experiment: {experiment_id}")
    
    return {"message": "Experiment deleted", "id": experiment_id}


# ============================================================
# WebSocket for Real-Time Metrics
# ============================================================

@router.websocket("/ws/{experiment_id}")
async def websocket_metrics(websocket: WebSocket, experiment_id: str):
    """
    WebSocket endpoint for real-time training metrics.
    
    Clients connect here to receive live updates during training.
    """
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connected for experiment: {experiment_id}")
    
    try:
        while True:
            # In production, this would receive actual metrics from Ray
            # For now, we just keep the connection alive
            data = await websocket.receive_text()
            
            # Echo back for testing
            await websocket.send_text(f"Received: {data}")
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected for experiment: {experiment_id}")


async def broadcast_metrics(experiment_id: str, metrics: MetricsUpdate):
    """Broadcast metrics to all connected WebSocket clients."""
    message = json.dumps({
        "experiment_id": experiment_id,
        "metrics": metrics.model_dump()
    })
    
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception:
            pass  # Client disconnected
