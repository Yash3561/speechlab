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
    wer: Optional[float] = None
    progress: float = 0.0
    worst_samples: Optional[List[dict]] = None



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


# Seed with demo data on startup
def seed_demo_data():
    """Seed demo experiments for UI testing."""
    from datetime import timedelta
    
    now = datetime.utcnow()
    
    demo_experiments = [
        {
            "id": "exp_001",
            "name": "whisper_tiny_librispeech",
            "status": "running",
            "created_at": (now - timedelta(hours=2)).isoformat() + "Z",
            "updated_at": (now - timedelta(minutes=15)).isoformat() + "Z",
            "current_epoch": 3,
            "total_epochs": 5,
            "train_loss": 0.55,
            "val_loss": 0.72,
            "wer": 5.2,
            "progress": 67,
        },
        {
            "id": "exp_002",
            "name": "whisper_base_noisy",
            "status": "completed",
            "created_at": (now - timedelta(days=1, hours=8)).isoformat() + "Z",
            "updated_at": (now - timedelta(hours=8)).isoformat() + "Z",
            "current_epoch": 10,
            "total_epochs": 10,
            "train_loss": 0.21,
            "val_loss": 0.34,
            "wer": 4.1,
            "progress": 100,
            "worst_samples": [
                {
                    "id": "sample_101",
                    "reference": "speech recognition is difficult",
                    "hypothesis": "peach wreck a nice beach is difficult",
                    "wer": 0.8,
                    "audio_url": "/api/audio/sample_101.wav"
                },
                {
                    "id": "sample_404",
                    "reference": "machine learning pipelines",
                    "hypothesis": "machine leaning pipe lines",
                    "wer": 0.4,
                    "audio_url": "/api/audio/sample_404.wav"
                },
                {
                    "id": "sample_202",
                    "reference": "artificial intelligence",
                    "hypothesis": "art official intelligence",
                    "wer": 0.6,
                    "audio_url": "/api/audio/sample_202.wav"
                },
                {
                    "id": "sample_303",
                    "reference": "neural networks are deep",
                    "hypothesis": "new role net works are deep",
                    "wer": 0.5,
                    "audio_url": "/api/audio/sample_303.wav"
                }

            ]
        },
        {
            "id": "exp_003",
            "name": "wav2vec2_baseline",
            "status": "failed",
            "created_at": (now - timedelta(days=3)).isoformat() + "Z",
            "updated_at": (now - timedelta(days=3, hours=-1)).isoformat() + "Z",
            "current_epoch": 2,
            "total_epochs": 10,
            "train_loss": 1.85,
            "val_loss": None,
            "wer": None,
            "progress": 34,
        },
        {
            "id": "exp_004",
            "name": "whisper_tiny_augmented",
            "status": "pending",
            "created_at": (now - timedelta(minutes=10)).isoformat() + "Z",
            "updated_at": (now - timedelta(minutes=10)).isoformat() + "Z",
            "current_epoch": 0,
            "total_epochs": 5,
            "train_loss": None,
            "val_loss": None,
            "wer": None,
            "progress": 0,
        },
    ]
    for exp in demo_experiments:
        experiments_db[exp["id"]] = exp


# Initialize demo data
seed_demo_data()


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

# Track connections per experiment
experiment_connections: dict[str, List[WebSocket]] = {}


@router.websocket("/ws/{experiment_id}")
async def websocket_metrics(websocket: WebSocket, experiment_id: str):
    """
    WebSocket endpoint for real-time training metrics.
    
    Sends simulated metrics every second for running experiments.
    In production, this would receive actual metrics from Ray Train.
    """
    await websocket.accept()
    
    # Track this connection
    if experiment_id not in experiment_connections:
        experiment_connections[experiment_id] = []
    experiment_connections[experiment_id].append(websocket)
    
    logger.info(f"WebSocket connected for experiment: {experiment_id}")
    
    try:
        # Simulation state
        step = 0
        train_loss = 2.5
        val_loss = 2.8
        
        while True:
            # Check if experiment is running
            exp = experiments_db.get(experiment_id)
            if not exp or exp.get("status") != "running":
                # Not running - just wait for status change
                await asyncio.sleep(1)
                # Send status update
                await websocket.send_json({
                    "type": "status",
                    "status": exp.get("status") if exp else "deleted",
                    "experiment_id": experiment_id
                })
                continue
            
            # Simulate training progress
            step += 10
            train_loss = max(0.05, train_loss - 0.02 + (asyncio.get_event_loop().time() % 0.01 - 0.005))
            val_loss = max(0.1, val_loss - 0.015 + (asyncio.get_event_loop().time() % 0.02 - 0.01))
            
            # Update experiment progress
            total_steps = exp.get("total_epochs", 5) * 1000
            progress = min(100, int((step / total_steps) * 100))
            current_epoch = min(exp.get("total_epochs", 5), step // 1000 + 1)
            
            experiments_db[experiment_id].update({
                "current_epoch": current_epoch,
                "progress": progress,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "updated_at": datetime.utcnow().isoformat(),
            })
            
            # Calculate simulated WER (improves as loss decreases)
            wer = max(3.0, 15.0 - (10.0 * (1 - train_loss / 2.5)))
            experiments_db[experiment_id]["wer"] = round(wer, 1)
            
            # Send metrics to client
            metrics = {
                "type": "metrics",
                "experiment_id": experiment_id,
                "data": {
                    "epoch": current_epoch,
                    "step": step,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "wer": round(wer, 1),
                    "learning_rate": 0.0001 * (0.95 ** current_epoch),
                    "gpu_util": 70 + (step % 30),
                    "throughput": 1000 + (step % 500),
                    "progress": progress,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            }
            
            await websocket.send_json(metrics)
            
            # Check if completed
            if progress >= 100:
                experiments_db[experiment_id]["status"] = "completed"
                experiments_db[experiment_id]["progress"] = 100
                await websocket.send_json({
                    "type": "completed",
                    "experiment_id": experiment_id,
                    "final_wer": experiments_db[experiment_id]["wer"]
                })
                logger.info(f"Experiment {experiment_id} completed!")
            
            # Wait before next update
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        if experiment_id in experiment_connections:
            experiment_connections[experiment_id].remove(websocket)
        logger.info(f"WebSocket disconnected for experiment: {experiment_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if experiment_id in experiment_connections and websocket in experiment_connections[experiment_id]:
            experiment_connections[experiment_id].remove(websocket)


async def broadcast_metrics(experiment_id: str, metrics: MetricsUpdate):
    """Broadcast metrics to all connected WebSocket clients for an experiment."""
    if experiment_id not in experiment_connections:
        return
        
    message = json.dumps({
        "type": "metrics",
        "experiment_id": experiment_id,
        "data": metrics.model_dump()
    })
    
    for connection in experiment_connections[experiment_id]:
        try:
            await connection.send_text(message)
        except Exception:
            pass  # Client disconnected
