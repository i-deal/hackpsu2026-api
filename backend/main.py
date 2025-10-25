"""
Main FastAPI application for Violence Detection Backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from config import settings
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Violence Detection API...")
    logger.info(f"📡 API running on {settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"🔧 Debug mode: {settings.API_DEBUG}")
    
    # Initialize components here
    # TODO: Initialize Redis connection
    # TODO: Initialize camera systems
    # TODO: Load ML models
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Violence Detection API...")
    # TODO: Cleanup resources


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    debug=settings.API_DEBUG,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Violence Detection API",
        "version": settings.API_VERSION,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api_version": settings.API_VERSION,
        "debug_mode": settings.API_DEBUG
    }


# Include API routes
# TODO: Import and include route modules
# from api.routes import detections, cameras, alerts, coordination, evidence, mapping, streaming
# app.include_router(detections.router, prefix="/api/v1/detections", tags=["detections"])
# app.include_router(cameras.router, prefix="/api/v1/cameras", tags=["cameras"])
# app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
# app.include_router(coordination.router, prefix="/api/v1/coordination", tags=["coordination"])
# app.include_router(evidence.router, prefix="/api/v1/evidence", tags=["evidence"])
# app.include_router(mapping.router, prefix="/api/v1/mapping", tags=["mapping"])
# app.include_router(streaming.router, prefix="/api/v1/streaming", tags=["streaming"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
