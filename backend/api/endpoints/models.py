"""
Models API Endpoints

Model registry operations and architecture management.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


# ============================================================
# Pydantic Models
# ============================================================

class ModelInfo(BaseModel):
    """Model registry entry."""
    id: str
    name: str
    architecture: str
    variant: str
    version: str
    stage: str  # development, staging, production
    created_at: str
    metrics: dict = {}
    tags: List[str] = []


class ModelArchitecture(BaseModel):
    """Available model architecture."""
    name: str
    variants: List[str]
    description: str
    parameters: dict = {}


# ============================================================
# Available Architectures
# ============================================================

ARCHITECTURES = {
    "whisper": ModelArchitecture(
        name="whisper",
        variants=["tiny", "base", "small"],
        description="OpenAI Whisper - Multilingual ASR",
        parameters={
            "tiny": {"params": "39M", "size": "150MB"},
            "base": {"params": "74M", "size": "290MB"},
            "small": {"params": "244M", "size": "970MB"},
        },
    ),
    "wav2vec2": ModelArchitecture(
        name="wav2vec2",
        variants=["base", "large"],
        description="Facebook Wav2Vec2 - Self-supervised ASR",
        parameters={
            "base": {"params": "95M", "size": "380MB"},
            "large": {"params": "317M", "size": "1.2GB"},
        },
    ),
}

# In-memory model registry
models_db: dict = {}


# ============================================================
# Endpoints
# ============================================================

@router.get("/architectures", response_model=List[ModelArchitecture])
async def list_architectures():
    """List available model architectures."""
    return list(ARCHITECTURES.values())


@router.get("/architectures/{name}", response_model=ModelArchitecture)
async def get_architecture(name: str):
    """Get details for a specific architecture."""
    if name not in ARCHITECTURES:
        raise HTTPException(status_code=404, detail="Architecture not found")
    return ARCHITECTURES[name]


@router.get("/", response_model=List[ModelInfo])
async def list_models():
    """List all registered models."""
    return list(models_db.values())


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get model details by ID."""
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    return models_db[model_id]


@router.post("/{model_id}/stage/{stage}")
async def update_model_stage(model_id: str, stage: str):
    """
    Update model stage (development → staging → production).
    """
    valid_stages = ["development", "staging", "production", "archived"]
    if stage not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage. Must be one of: {valid_stages}"
        )
    
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    models_db[model_id]["stage"] = stage
    
    return {"message": f"Model {model_id} moved to {stage}"}
