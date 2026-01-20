"""
Inference API Endpoints

Handle real-time transcription requests.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import shutil
import os
import time
import random 
from backend.core.config import settings

router = APIRouter()

class TranscriptionResponse(BaseModel):
    text: str
    confidence: float
    latency_ms: float
    model_id: str

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model_id: str = Form(...)
):
    """
    Transcribe an uploaded audio file.
    """
    start_time = time.time()
    
    # Save temp file
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = f"{temp_dir}/{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # TODO: Load actual model and transcribe
        # For demo, we simulate transcription based on delay
        
        # Simulate processing time
        time.sleep(1.5)
        
        # Mock responses for demo
        mock_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Speech recognition is a challenging but rewarding field.",
            "Machine learning pipelines require robust infrastructure.",
            "Apple's ecosystem provides a seamless user experience.",
            "Regression testing ensures that new changes do not break existing functionality."
        ]
        
        text = random.choice(mock_texts)
        if "error" in file.filename.lower():
            text = "This is a simulated error where the model failed to understand the audio."
        
        latency = (time.time() - start_time) * 1000
        
        return TranscriptionResponse(
            text=text,
            confidence=0.95 + (random.random() * 0.04),
            latency_ms=round(latency, 2),
            model_id=model_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
