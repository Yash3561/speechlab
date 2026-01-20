"""
SpeechLab Utility Functions

Common utilities used across the application.
"""

import hashlib
import time
from pathlib import Path
from typing import Union
from datetime import datetime


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat()


def generate_run_id(prefix: str = "run") -> str:
    """
    Generate a unique run ID.
    
    Args:
        prefix: Prefix for the run ID
        
    Returns:
        Unique run ID like 'run_20240119_143052_abc123'
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    hash_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    return f"{prefix}_{timestamp}_{hash_suffix}"


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like '2h 30m 15s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def calculate_audio_hash(filepath: Union[str, Path]) -> str:
    """
    Calculate MD5 hash of an audio file for deduplication.
    
    Args:
        filepath: Path to audio file
        
    Returns:
        MD5 hash string
    """
    filepath = Path(filepath)
    hasher = hashlib.md5()
    
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
            
    return hasher.hexdigest()
