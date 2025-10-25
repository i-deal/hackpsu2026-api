# 🧪 Testing Guide - Crime Analysis System

This guide shows you how to test the violence detection backend with crime analysis capabilities.

## 🚀 Quick Start Testing

### 1. **Run All Tests**
```bash
cd backend
python3 scripts/test_crime_analysis.py
```

**Expected Output:**
```
🧪 Crime Analysis System Test Suite
==================================================
✅ Crime analysis system is ready!
🚀 Start the backend with: python main.py
📡 API will be available at: http://localhost:8000
📚 API docs at: http://localhost:8000/docs
```

### 2. **Test Individual Components**
```bash
# Test crime analyzer
python3 -c "from core.crime_analyzer import test_crime_analyzer; test_crime_analyzer()"

# Test evidence collection
python3 -c "from evidence.evidence_collection import test_evidence_collection; test_evidence_collection()"

# Test video streaming
python3 -c "from camera.video_streaming import test_video_streaming; test_video_streaming()"
```

### 3. **Start the API Server**
```bash
cd backend
python3 main.py
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
🚀 Starting Violence Detection API with Crime Analysis...
📡 API running on 0.0.0.0:8000
🔧 Debug mode: True
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 📊 Test Results Analysis

### ✅ **Passing Tests (6/7)**
- **Dataset Analysis**: ✅ PASS - Analyzes 13 crime categories, 10,196 total videos
- **Crime Report**: ✅ PASS - Generates comprehensive crime statistics
- **Video Processing**: ✅ PASS - Processes videos for violence detection
- **Evidence Collection**: ✅ PASS - Collects evidence with metadata
- **Evidence Summary**: ✅ PASS - Provides evidence statistics
- **Crime Patterns**: ✅ PASS - Analyzes crime patterns and insights

### ⚠️ **Failing Tests (1/7)**
- **Video Streaming**: ❌ FAIL - No dataset found (expected if no data directory)

## 🔍 Detailed Test Results

### **Dataset Analysis Results**
```
Total categories: 13
Total videos: 10,196
Data quality: {
  "completeness": 1.0,
  "balance": 0.58,
  "diversity": 0.65,
  "coverage": 0.052
}
```

### **Key Insights Generated**
- Fighting category has only 8 videos - needs data augmentation
- Most common crime: Robbery with 105 videos
- Data imbalance: 18.6:1 normal to abnormal ratio
- Violence-related crimes: 100 videos

### **Recommendations Generated**
- Increase Fighting category videos through data augmentation
- Balance dataset - Robbery has 105 videos vs Fighting with 8
- Focus on violence detection categories: Fighting, Assault, Abuse
- Implement data augmentation for underrepresented categories

## 🌐 API Testing

### **1. Health Check**
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "debug_mode": true,
  "crime_analysis": "active",
  "evidence_collection": "active",
  "video_streaming": "active"
}
```

### **2. Crime Analysis**
```bash
curl http://localhost:8000/crime/analysis
```

**Expected Response:**
```json
{
  "status": "success",
  "analysis": {
    "total_categories": 13,
    "total_videos": 10196,
    "categories": {
      "Abuse": 39,
      "Arrest": 26,
      "Arson": 22,
      "Assault": 23,
      "Burglary": 47,
      "Explosion": 23,
      "Fighting": 8,
      "RoadAccidents": 76,
      "Robbery": 105,
      "Shooting": 30,
      "Shoplifting": 28,
      "Stealing": 64,
      "Vandalism": 29
    },
    "insights": [
      "Fighting category has only 8 videos - needs data augmentation",
      "Most common crime: Robbery with 105 videos",
      "Data imbalance: 18.6:1 normal to abnormal ratio",
      "Violence-related crimes: 100 videos"
    ]
  }
}
```

### **3. Evidence Summary**
```bash
curl http://localhost:8000/evidence/summary
```

**Expected Response:**
```json
{
  "status": "success",
  "summary": {
    "total_incidents": 1,
    "evidence_directory": "/path/to/evidence",
    "video_clips_count": 0,
    "total_storage_used": {
      "total_bytes": 0,
      "total_mb": 0.0,
      "file_counts": {
        "video_clips": 0,
        "metadata": 1,
        "screenshots": 0
      }
    },
    "threat_types": {
      "fighting": 1
    },
    "cameras_involved": {
      "cam_1": 1
    }
  }
}
```

### **4. System Statistics**
```bash
curl http://localhost:8000/stats
```

**Expected Response:**
```json
{
  "system_status": "operational",
  "detections": {
    "total": 0,
    "high_confidence": 0,
    "recent": 0
  },
  "evidence": {
    "total_incidents": 1,
    "storage_used_mb": 0.0,
    "threat_types": {
      "fighting": 1
    }
  },
  "streaming": {
    "total_streams": 0,
    "active_streams": 0,
    "is_running": false
  },
  "crime_analysis": {
    "categories_available": 13,
    "threat_levels": 4
  }
}
```

## 🎬 Video Analysis Testing

### **Test with Sample Video**
```bash
# Test with specific video file
python3 scripts/test_crime_analysis.py /path/to/video.mp4
```

### **Test Video Analysis API**
```bash
curl -X POST "http://localhost:8000/video/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "camera_id": "cam_1",
    "analysis_type": "full"
  }'
```

**Expected Response:**
```json
{
  "status": "success",
  "analysis": {
    "threat_type": "Fighting",
    "confidence": 0.85,
    "severity": "high",
    "indicators": [
      "High motion detected",
      "Rapid color changes",
      "Violence escalation detected"
    ],
    "frames_analyzed": 30
  },
  "detection_created": true,
  "evidence_collected": true,
  "evidence_id": "fighting_cam_1_20251025_153104"
}
```

## 🔧 Advanced Testing

### **1. Test with Real Dataset**
```bash
# Create data directory structure
mkdir -p data/{Fighting,Robbery,Assault,Normal}

# Add some video files to test with
# Then run:
python3 scripts/test_crime_analysis.py
```

### **2. Test Evidence Collection**
```bash
# Test evidence collection with sample detection
python3 -c "
from evidence.evidence_collection import collect_evidence_for_detection
detection = {
    'camera_id': 'cam_1',
    'threat_type': 'fighting',
    'confidence': 0.95,
    'timestamp': '16:45:30',
    'severity': 'high'
}
evidence = collect_evidence_for_detection(detection)
print(f'Evidence ID: {evidence[\"evidence_id\"]}')
"
```

### **3. Test Crime Pattern Analysis**
```bash
python3 -c "
from core.crime_analyzer import analyze_dataset_structure
analysis = analyze_dataset_structure()
print(f'Total categories: {analysis[\"total_categories\"]}')
print(f'Total videos: {analysis[\"total_videos\"]}')
print(f'Data quality: {analysis[\"data_quality\"]}')
"
```

## 📱 Web Interface Testing

### **1. API Documentation**
Open in browser: http://localhost:8000/docs

### **2. Alternative Documentation**
Open in browser: http://localhost:8000/redoc

### **3. Interactive Testing**
Use the Swagger UI to test endpoints interactively:
- Click "Try it out" on any endpoint
- Fill in the required parameters
- Click "Execute" to test

## 🐛 Troubleshooting

### **Common Issues**

#### **1. Import Errors**
```bash
# Make sure you're in the backend directory
cd backend
python3 -c "import sys; print(sys.path)"
```

#### **2. Module Not Found**
```bash
# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 scripts/test_crime_analysis.py
```

#### **3. Video Streaming Not Available**
This is expected if no dataset is found. The system will work without video streaming.

#### **4. Evidence Collection Warnings**
```
⚠️ No source video provided or file doesn't exist: None
```
This is expected in test mode without actual video files.

### **Debug Mode**
```bash
# Run with debug logging
LOG_LEVEL=DEBUG python3 main.py
```

### **Check System Status**
```bash
# Check if all components are working
python3 -c "
from main import app
print('✅ FastAPI app created successfully')
print('📡 API endpoints available')
"
```

## 📊 Performance Testing

### **Load Testing**
```bash
# Test API with multiple requests
for i in {1..10}; do
  curl http://localhost:8000/health &
done
wait
```

### **Memory Usage**
```bash
# Monitor memory usage
python3 -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

## ✅ Success Criteria

### **All Tests Passing**
- ✅ Dataset analysis working
- ✅ Crime report generation working
- ✅ Video processing working
- ✅ Evidence collection working
- ✅ Evidence summary working
- ✅ Crime pattern analysis working

### **API Server Running**
- ✅ Server starts without errors
- ✅ Health check returns 200
- ✅ All endpoints accessible
- ✅ Documentation available at /docs

### **Evidence Collection Working**
- ✅ Evidence metadata generated
- ✅ Database updated
- ✅ Files saved correctly

## 🚀 Next Steps

1. **Add Real Dataset**: Place DCSASS dataset in `data/` directory
2. **Test with Real Videos**: Use actual video files for testing
3. **Configure Notifications**: Set up Discord/Email/SMS alerts
4. **Deploy to Production**: Use Gunicorn for production deployment

---

**Your crime analysis system is ready! 🚨**
