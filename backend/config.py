"""
Configuration management for Violence Detection Backend
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = True
    API_TITLE: str = "Violence Detection API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Real-time violence detection and multi-camera coordination system"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 0
    
    # Database Configuration (optional)
    DATABASE_URL: Optional[str] = "sqlite:///./violence_detection.db"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    EVIDENCE_DIR: Path = BASE_DIR / "evidence"
    MODEL_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "data" / "logs"
    TEMP_DIR: Path = BASE_DIR / "data" / "temp"
    CACHE_DIR: Path = BASE_DIR / "data" / "cache"
    
    # Model Paths
    CNN_MODEL_PATH: str = "./models/violence_model_best.pth"
    MULTI_CLASS_MODEL_PATH: str = "./models/multi_class_violence_model_best.pth"
    
    # Processing Configuration
    MAX_FRAMES_PER_VIDEO: int = 30
    CLIP_DURATION_BEFORE: int = 10  # seconds before incident
    CLIP_DURATION_AFTER: int = 10   # seconds after incident
    CONFIDENCE_THRESHOLD: float = 0.7
    VIOLENCE_THRESHOLD: float = 0.8
    
    # Camera Configuration
    MAX_CAMERAS: int = 10
    CAMERA_TIMEOUT: int = 30  # seconds
    STREAM_FPS: int = 30
    STREAM_RESOLUTION: tuple = (640, 480)
    
    # Notification Settings
    DISCORD_WEBHOOK_URL: Optional[str] = None
    EMAIL_SMTP_SERVER: Optional[str] = None
    EMAIL_SMTP_PORT: int = 587
    EMAIL_FROM: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    
    # Security (optional)
    API_KEY: Optional[str] = None
    JWT_SECRET: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        settings.DATA_DIR,
        settings.EVIDENCE_DIR,
        settings.MODEL_DIR,
        settings.LOGS_DIR,
        settings.TEMP_DIR,
        settings.CACHE_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()
