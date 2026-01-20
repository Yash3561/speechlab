"""
Training API Endpoints

API endpoints for starting and managing training jobs.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
import asyncio

from backend.core.logging import logger
from backend.core.utils import generate_run_id

router = APIRouter()


# ============================================================
# Pydantic Models
# ============================================================

class TrainingJobConfig(BaseModel):
    """Configuration for a training job."""
    experiment_name: str
    model_architecture: str = "whisper"
    model_variant: str = "tiny"
    batch_size: int = 8
    learning_rate: float = 0.0001
    max_epochs: int = 5
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None


class TrainingJobStatus(BaseModel):
    """Status of a training job."""
    job_id: str
    experiment_name: str
    status: str  # queued, running, completed, failed
    current_epoch: int
    total_epochs: int
    train_loss: Optional[float]
    val_loss: Optional[float]
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


# ============================================================
# In-Memory Job Storage
# ============================================================

training_jobs: dict = {}


# ============================================================
# Background Training Task
# ============================================================

async def run_training_job(job_id: str, config: TrainingJobConfig):
    """
    Background task that runs a training job.
    
    This is a simplified version that simulates training.
    In production, this would use Ray Train.
    """
    try:
        job = training_jobs[job_id]
        job["status"] = "running"
        job["started_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Starting training job: {job_id}")
        
        # Simulate training epochs
        for epoch in range(1, config.max_epochs + 1):
            if job["status"] == "stopped":
                logger.info(f"Training job {job_id} stopped by user")
                break
            
            # Simulate training time
            await asyncio.sleep(2)
            
            # Simulate metrics
            train_loss = 2.5 - (epoch * 0.4) + (0.1 * (epoch % 2))
            val_loss = train_loss + 0.2
            
            job["current_epoch"] = epoch
            job["train_loss"] = round(train_loss, 4)
            job["val_loss"] = round(val_loss, 4)
            
            logger.info(
                f"Job {job_id} - Epoch {epoch}/{config.max_epochs} - "
                f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )
        
        # Mark as completed
        if job["status"] != "stopped":
            job["status"] = "completed"
        job["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Training job {job_id} completed")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)


# ============================================================
# REST Endpoints
# ============================================================

@router.get("/jobs", response_model=list[TrainingJobStatus])
async def list_training_jobs():
    """List all training jobs."""
    return list(training_jobs.values())


@router.post("/jobs", response_model=TrainingJobStatus)
async def create_training_job(
    config: TrainingJobConfig,
    background_tasks: BackgroundTasks,
):
    """
    Create and start a new training job.
    
    This queues the job for execution in the background.
    """
    job_id = generate_run_id("train")
    
    job = {
        "job_id": job_id,
        "experiment_name": config.experiment_name,
        "status": "queued",
        "current_epoch": 0,
        "total_epochs": config.max_epochs,
        "train_loss": None,
        "val_loss": None,
        "started_at": None,
        "completed_at": None,
        "error": None,
    }
    
    training_jobs[job_id] = job
    
    # Queue background training
    background_tasks.add_task(run_training_job, job_id, config)
    
    logger.info(f"Created training job: {job_id}")
    
    return job


@router.get("/jobs/{job_id}", response_model=TrainingJobStatus)
async def get_training_job(job_id: str):
    """Get status of a specific training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    return training_jobs[job_id]


@router.post("/jobs/{job_id}/stop")
async def stop_training_job(job_id: str):
    """Stop a running training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    if job["status"] != "running":
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot stop job with status: {job['status']}"
        )
    
    job["status"] = "stopped"
    logger.info(f"Stopping training job: {job_id}")
    
    return {"message": "Job stop requested", "job_id": job_id}


@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: str):
    """Delete a training job (only if not running)."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    if job["status"] == "running":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete running job. Stop it first."
        )
    
    del training_jobs[job_id]
    logger.info(f"Deleted training job: {job_id}")
    
    return {"message": "Job deleted", "job_id": job_id}
