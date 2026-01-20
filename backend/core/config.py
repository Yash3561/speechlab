"""
SpeechLab Configuration System

Pydantic-based settings with environment variable support.
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    
    # Database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/speechlab",
        alias="DATABASE_URL"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379",
        alias="REDIS_URL"
    )
    
    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        alias="MLFLOW_TRACKING_URI"
    )
    
    # S3/MinIO Storage
    s3_endpoint_url: str = Field(
        default="http://localhost:9000",
        alias="S3_ENDPOINT_URL"
    )
    s3_access_key: str = Field(default="minioadmin", alias="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="minioadmin", alias="S3_SECRET_KEY")
    s3_bucket: str = Field(default="speechlab-artifacts", alias="S3_BUCKET")
    
    # Ray
    ray_address: Optional[str] = Field(default=None, alias="RAY_ADDRESS")
    
    # Audio Processing
    sample_rate: int = 16000
    max_audio_duration: float = 30.0  # seconds
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Export settings instance
settings = get_settings()
