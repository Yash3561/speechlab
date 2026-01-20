"""
MLflow API Endpoints

API endpoints for accessing MLflow experiment tracking data.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime

from backend.core.logging import logger

router = APIRouter()


# ============================================================
# Pydantic Models
# ============================================================

class RunMetrics(BaseModel):
    """Metrics from an MLflow run."""
    run_id: str
    run_name: Optional[str]
    status: str
    start_time: Optional[int]
    metrics: dict


class RunComparison(BaseModel):
    """Comparison of multiple runs."""
    runs: dict


class ModelVersion(BaseModel):
    """Registered model version."""
    name: str
    version: str
    stage: str
    status: str


# ============================================================
# In-Memory Mock Storage (for when MLflow is not available)
# ============================================================

mock_runs = {
    "run_001": {
        "run_id": "run_001",
        "run_name": "whisper_tiny_v1",
        "status": "FINISHED",
        "start_time": int(datetime.utcnow().timestamp() * 1000) - 86400000,
        "metrics": {
            "train_loss": 0.45,
            "val_loss": 0.52,
            "wer": 5.2,
            "epoch": 10,
        }
    },
    "run_002": {
        "run_id": "run_002", 
        "run_name": "whisper_base_v1",
        "status": "FINISHED",
        "start_time": int(datetime.utcnow().timestamp() * 1000) - 172800000,
        "metrics": {
            "train_loss": 0.32,
            "val_loss": 0.41,
            "wer": 4.1,
            "epoch": 15,
        }
    },
    "run_003": {
        "run_id": "run_003",
        "run_name": "wav2vec2_baseline",
        "status": "FAILED",
        "start_time": int(datetime.utcnow().timestamp() * 1000) - 259200000,
        "metrics": {
            "train_loss": 1.85,
            "epoch": 2,
        }
    },
}

mock_models = [
    {
        "name": "whisper-tiny-asr",
        "latest_versions": [
            {"version": "3", "stage": "Production", "status": "READY"},
            {"version": "2", "stage": "Staging", "status": "READY"},
            {"version": "1", "stage": "Archived", "status": "READY"},
        ]
    },
    {
        "name": "whisper-base-asr",
        "latest_versions": [
            {"version": "1", "stage": "Staging", "status": "READY"},
        ]
    },
]


# ============================================================
# REST Endpoints
# ============================================================

@router.get("/runs", response_model=List[RunMetrics])
async def list_runs(
    experiment_name: str = "speechlab",
    max_results: int = 100,
):
    """List all runs in an experiment."""
    try:
        from backend.tracking import get_tracker
        tracker = get_tracker(experiment_name)
        runs = tracker.list_runs(max_results=max_results)
        
        if not runs:
            # Return mock data if no real runs
            return list(mock_runs.values())
        
        return runs
    except Exception as e:
        logger.warning(f"Using mock runs: {e}")
        return list(mock_runs.values())


@router.get("/runs/{run_id}", response_model=RunMetrics)
async def get_run(run_id: str):
    """Get details of a specific run."""
    try:
        from backend.tracking import get_tracker
        tracker = get_tracker()
        run = tracker.get_run(run_id)
        
        if run:
            return run
    except Exception:
        pass
    
    # Fall back to mock
    if run_id in mock_runs:
        return mock_runs[run_id]
    
    raise HTTPException(status_code=404, detail="Run not found")


@router.get("/runs/compare")
async def compare_runs(
    run_ids: str,  # Comma-separated
    metrics: Optional[str] = None,  # Comma-separated
):
    """Compare metrics across multiple runs."""
    ids = [r.strip() for r in run_ids.split(",")]
    metric_list = [m.strip() for m in metrics.split(",")] if metrics else None
    
    try:
        from backend.tracking import get_tracker
        tracker = get_tracker()
        comparison = tracker.compare_runs(ids, metric_list)
        
        if comparison:
            return {"runs": comparison}
    except Exception:
        pass
    
    # Fall back to mock comparison
    comparison = {}
    for run_id in ids:
        if run_id in mock_runs:
            if metric_list:
                comparison[run_id] = {
                    k: v for k, v in mock_runs[run_id]["metrics"].items()
                    if k in metric_list
                }
            else:
                comparison[run_id] = mock_runs[run_id]["metrics"]
    
    return {"runs": comparison}


@router.get("/runs/best")
async def get_best_run(
    metric: str = "val_loss",
    ascending: bool = True,
):
    """Get the best run based on a metric."""
    try:
        from backend.tracking import get_tracker
        tracker = get_tracker()
        best = tracker.get_best_run(metric, ascending)
        
        if best:
            return best
    except Exception:
        pass
    
    # Fall back to mock
    runs = list(mock_runs.values())
    runs_with_metric = [r for r in runs if metric in r.get("metrics", {})]
    
    if not runs_with_metric:
        raise HTTPException(status_code=404, detail=f"No runs with metric '{metric}'")
    
    sorted_runs = sorted(
        runs_with_metric,
        key=lambda r: r["metrics"].get(metric, float("inf")),
        reverse=not ascending,
    )
    
    return sorted_runs[0]


@router.get("/models")
async def list_models():
    """List all registered models."""
    try:
        from backend.tracking import get_registry
        registry = get_registry()
        models = registry.list_models()
        
        if models:
            return models
    except Exception:
        pass
    
    return mock_models


@router.post("/models/{model_name}/transition")
async def transition_model(
    model_name: str,
    version: str,
    stage: str,
):
    """Transition a model version to a new stage."""
    if stage not in ["Staging", "Production", "Archived"]:
        raise HTTPException(status_code=400, detail="Invalid stage")
    
    try:
        from backend.tracking import get_registry
        registry = get_registry()
        success = registry.transition_model_stage(model_name, version, stage)
        
        if success:
            return {"message": f"Model {model_name} v{version} transitioned to {stage}"}
    except Exception as e:
        logger.error(f"Failed to transition model: {e}")
    
    return {"message": f"Model {model_name} v{version} transitioned to {stage} (mock)"}
