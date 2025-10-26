"""
Main FastAPI application for Violence Detection Backend
Integrated with crime analysis patterns from DCSASS dataset
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import cv2
import io
import json
from datetime import datetime
from contextlib import asynccontextmanager

from config import settings
from utils.logger import setup_logger
from utils.constants import MULTI_CLASS_CATEGORIES, THREAT_LEVELS

# Import crime analysis modules
from core.crime_analyzer import crime_analyzer, analyze_dataset_structure, generate_crime_report
from core.real_video_processor import video_processor, process_video_file, process_dataset_directory
from evidence.evidence_collection import evidence_collector, collect_evidence_for_detection, get_evidence_summary
from camera.video_streaming import video_streaming, setup_video_streams, get_camera_frame, update_camera_threat

# Setup logging
logger = setup_logger(__name__)

# Pydantic models
class Detection(BaseModel):
    camera_id: str
    threat_type: str
    confidence: float
    timestamp: str
    location: dict
    severity: Optional[str] = "medium"
    indicators: Optional[List[str]] = []

class Alert(BaseModel):
    incident_id: str
    threat_type: str
    severity: str
    cameras: List[str]
    timestamp: str
    confidence: float

class VideoAnalysisRequest(BaseModel):
    video_path: str
    camera_id: Optional[str] = None
    analysis_type: Optional[str] = "full"  # "full", "quick", "crime_focused"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Violence Detection API with Crime Analysis...")
    logger.info(f"📡 API running on {settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"🔧 Debug mode: {settings.API_DEBUG}")
    
    # Initialize crime analysis
    logger.info("🔍 Initializing crime analysis system...")
    try:
        # Analyze dataset structure if available
        if settings.DATA_DIR.exists():
            dataset_analysis = analyze_dataset_structure(str(settings.DATA_DIR))
            logger.info(f"📊 Dataset analysis: {dataset_analysis['total_categories']} categories, {dataset_analysis['total_videos']} videos")
        
        # Setup video streaming
        if setup_video_streams():
            logger.info("📹 Video streaming system initialized")
        else:
            logger.warning("⚠️ Video streaming system not available")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Violence Detection API...")

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

# Global state
detections = []
alerts = []
analysis_results = {}

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Violence Detection API with Crime Analysis",
        "version": settings.API_VERSION,
        "status": "running",
        "features": [
            "Real-time crime analysis",
            "DCSASS dataset integration",
            "Evidence collection",
            "Video streaming",
            "Multi-camera coordination"
        ],
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api_version": settings.API_VERSION,
        "debug_mode": settings.API_DEBUG,
        "crime_analysis": "active",
        "evidence_collection": "active",
        "video_streaming": "active"
    }

# Crime Analysis Endpoints
@app.get("/crime/analysis")
async def get_crime_analysis():
    """Get crime analysis data"""
    try:
        analysis = analyze_dataset_structure()
        return {"status": "success", "analysis": analysis}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/crime/report")
async def get_crime_report():
    """Get comprehensive crime report"""
    try:
        report = generate_crime_report()
        return {"status": "success", "report": report}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/crime/categories")
async def get_crime_categories():
    """Get available crime categories"""
    return {
        "status": "success",
        "categories": MULTI_CLASS_CATEGORIES,
        "threat_levels": THREAT_LEVELS
    }

# Video Processing Endpoints
@app.post("/video/analyze")
async def analyze_video(request: VideoAnalysisRequest):
    """Analyze a video for crime activity"""
    try:
        result = process_video_file(request.video_path)
        
        if result.get("status") == "success":
            # Create detection if threat detected
            if result.get("confidence", 0) > settings.CONFIDENCE_THRESHOLD:
                detection = Detection(
                    camera_id=request.camera_id or "unknown",
                    threat_type=result["threat_type"],
                    confidence=result["confidence"],
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    location={"x": 150, "y": 200},
                    severity=result["severity"],
                    indicators=result["indicators"]
                )
                detections.append(detection)
                
                # Collect evidence
                evidence_metadata = collect_evidence_for_detection(
                    detection.dict(), 
                    request.video_path
                )
                
                # Update camera threat overlay
                if request.camera_id:
                    update_camera_threat(request.camera_id, detection.dict())
                
                return {
                    "status": "success",
                    "analysis": result,
                    "detection_created": True,
                    "evidence_collected": evidence_metadata is not None,
                    "evidence_id": evidence_metadata.get("evidence_id") if evidence_metadata else None
                }
            else:
                return {
                    "status": "success",
                    "analysis": result,
                    "detection_created": False,
                    "message": "No threat detected above confidence threshold"
                }
        else:
            return {"status": "error", "message": result.get("message", "Analysis failed")}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/video/process-dataset")
async def process_dataset(dataset_path: str):
    """Process entire dataset for crime analysis"""
    try:
        results = process_dataset_directory(dataset_path)
        
        # Save results
        video_processor.save_analysis_results(results)
        
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Detection Endpoints
@app.get("/detections")
async def get_detections():
    """Get all recent detections"""
    return {"detections": detections[-20:]}  # Last 20 detections

@app.post("/detection")
async def add_detection(detection: Detection):
    """Add a new detection"""
    detections.append(detection)
    
    # Collect evidence for high-confidence detections
    evidence_metadata = None
    if detection.confidence > settings.CONFIDENCE_THRESHOLD:
        evidence_metadata = collect_evidence_for_detection(detection.dict())
    
    # Update camera threat overlay
    update_camera_threat(detection.camera_id, detection.dict())
    
    return {
        "message": "Detection added",
        "detection_id": len(detections),
        "evidence_collected": evidence_metadata is not None,
        "evidence_id": evidence_metadata.get("evidence_id") if evidence_metadata else None
    }

@app.get("/detections/stats")
async def get_detection_stats():
    """Get detection statistics"""
    if not detections:
        return {"total_detections": 0, "threat_types": {}, "confidence_stats": {}}
    
    # Count threat types
    threat_types = {}
    for detection in detections:
        threat_type = detection.threat_type
        threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
    
    # Confidence statistics
    confidences = [d.confidence for d in detections]
    confidence_stats = {
        "average": sum(confidences) / len(confidences),
        "max": max(confidences),
        "min": min(confidences)
    }
    
    return {
        "total_detections": len(detections),
        "threat_types": threat_types,
        "confidence_stats": confidence_stats,
        "recent_detections": len([d for d in detections if d.confidence > settings.CONFIDENCE_THRESHOLD])
    }

# Evidence Collection Endpoints
@app.get("/evidence/summary")
async def get_evidence_summary_endpoint():
    """Get evidence collection summary"""
    try:
        from evidence.evidence_collection import get_evidence_summary
        summary = get_evidence_summary()
        return {"status": "success", "summary": summary}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/evidence/files")
async def list_evidence_files_endpoint():
    """List all evidence files"""
    try:
        from evidence.evidence_collection import list_evidence_files
        files = list_evidence_files()
        return {"status": "success", "files": files}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/evidence/{evidence_id}")
async def get_evidence_endpoint(evidence_id: str):
    """Get specific evidence by ID"""
    try:
        from evidence.evidence_collection import get_evidence_by_id
        evidence = get_evidence_by_id(evidence_id)
        if evidence:
            return {"status": "success", "evidence": evidence}
        else:
            return {"status": "error", "message": "Evidence not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Video Streaming Endpoints
@app.get("/streaming/status")
async def get_streaming_status():
    """Get video streaming status"""
    try:
        status = video_streaming.get_stream_status()
        return {"status": "success", "streaming_status": status}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/streaming/camera/{camera_id}/frame")
async def get_camera_frame(camera_id: str):
    """Get current frame from camera"""
    try:
        frame_base64 = get_camera_frame(camera_id)
        if frame_base64:
            return {"status": "success", "frame": frame_base64}
        else:
            return {"status": "error", "message": "No frame available"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/streaming/camera/{camera_id}/frame/image")
async def get_camera_frame_image(camera_id: str):
    """Get current frame as image"""
    try:
        frame_base64 = get_camera_frame(camera_id)
        if frame_base64:
            import base64
            frame_bytes = base64.b64decode(frame_base64)
            return StreamingResponse(
                io.BytesIO(frame_bytes),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            raise HTTPException(status_code=404, detail="No frame available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# System Statistics
@app.get("/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        # Get evidence summary
        evidence_summary = get_evidence_summary()
        
        # Get streaming status
        streaming_status = video_streaming.get_stream_status()
        
        # Calculate detection stats
        high_confidence_detections = len([d for d in detections if d.confidence > settings.CONFIDENCE_THRESHOLD])
        
        return {
            "system_status": "operational",
            "detections": {
                "total": len(detections),
                "high_confidence": high_confidence_detections,
                "recent": len(detections[-10:])
            },
            "evidence": {
                "total_incidents": evidence_summary.get("total_incidents", 0),
                "storage_used_mb": evidence_summary.get("total_storage_used", {}).get("total_mb", 0),
                "threat_types": evidence_summary.get("threat_types", {})
            },
            "streaming": {
                "total_streams": streaming_status.get("total_streams", 0),
                "active_streams": streaming_status.get("active_streams", 0),
                "is_running": streaming_status.get("is_running", False)
            },
            "crime_analysis": {
                "categories_available": len(MULTI_CLASS_CATEGORIES),
                "threat_levels": len(THREAT_LEVELS)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

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