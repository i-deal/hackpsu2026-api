# 🚨 Violence Detection Backend

A comprehensive real-time violence detection system with multi-camera coordination, evidence collection, and building mapping capabilities.

## 🏗️ Architecture

```
backend/
├── core/                    # Core processing modules
├── models/                  # ML model definitions
├── camera/                  # Camera management
├── coordination/            # Multi-camera coordination
├── evidence/               # Evidence collection
├── mapping/                # Building map integration
├── api/                    # API routes
├── utils/                  # Utilities and helpers
├── notifications/          # Notification systems
├── tests/                  # Test suite
└── scripts/               # Utility scripts
```

## 🚀 Quick Start

### 1. Installation

```bash
# Run the automated installation script
./scripts/install_backend.sh

# Or install manually
python3 -m venv venv
source venv/bin/activate
pip install -r ../Requirements.txt
```

### 2. Configuration

```bash
# Copy and configure environment file
cp .env.example .env
# Edit .env with your settings
```

### 3. Start Redis

```bash
# macOS
brew install redis
redis-server

# Ubuntu/Debian
sudo apt-get install redis-server
redis-server
```

### 4. Run the Backend

```bash
# Development
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📋 Features

### 🎥 Multi-Camera System
- Real-time video streaming
- Camera coordination and management
- Automatic failover and reconnection

### 🤖 AI-Powered Detection
- Binary violence detection (CNN)
- Multi-class violence categorization (13 categories)
- Real-time frame analysis
- Confidence-based filtering

### 📊 Evidence Collection
- Automatic video clip extraction
- Metadata management
- Evidence retention policies
- Secure storage

### 🗺️ Building Mapping
- Floor plan integration
- Threat location tracking
- Zone-based management
- Incident mapping

### 🔔 Notification System
- Discord webhook integration
- Email alerts (SMTP)
- SMS notifications (Twilio)
- Multi-priority alerting

### 🔄 Coordination
- Multi-camera protocol (MCP)
- Redis-based coordination
- Incident tracking
- Cross-camera correlation

## 🛠️ Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_video_processing.py

# Run with coverage
python -m pytest --cov=. tests/
```

### System Testing

```bash
# Test system components
python scripts/test_system.py

# Test specific modules
python -c "from core.video_processor import VideoProcessor; print('✅ Core module OK')"
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 📁 Directory Structure

### Core Modules (`core/`)
- `real_video_processor.py` - Main video processing pipeline
- `cnn_inference.py` - Binary violence detection
- `multi_class_inference.py` - Multi-class classification
- `video_analysis.py` - Frame analysis utilities

### Camera Management (`camera/`)
- `multi_camera_system.py` - Multi-camera coordination
- `video_streaming.py` - Video streaming system
- `video_streamer.py` - Real-time streaming
- `camera_manager.py` - Camera configuration

### Coordination (`coordination/`)
- `mcp_coordination.py` - Multi-camera protocol hub
- `redis_mcp_coordination.py` - Redis-based coordination
- `incident_tracker.py` - Incident tracking

### Evidence Collection (`evidence/`)
- `evidence_collection.py` - Evidence collector
- `video_clipper.py` - Video clip extraction
- `metadata_manager.py` - Evidence metadata

### Building Mapping (`mapping/`)
- `building_map.py` - Building map system
- `threat_locator.py` - Threat location tracking
- `zone_manager.py` - Zone management

### API Routes (`api/routes/`)
- `detections.py` - Detection endpoints
- `cameras.py` - Camera endpoints
- `alerts.py` - Alert endpoints
- `coordination.py` - Coordination endpoints
- `evidence.py` - Evidence endpoints
- `mapping.py` - Map endpoints
- `streaming.py` - Streaming endpoints

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Models
CNN_MODEL_PATH=./models/violence_model_best.pth
MULTI_CLASS_MODEL_PATH=./models/multi_class_violence_model_best.pth

# Processing
CONFIDENCE_THRESHOLD=0.7
VIOLENCE_THRESHOLD=0.8

# Notifications
DISCORD_WEBHOOK_URL=your_webhook_url
EMAIL_SMTP_SERVER=smtp.gmail.com
TWILIO_ACCOUNT_SID=your_account_sid
```

## 📊 API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - API documentation

### Detection Endpoints
- `POST /api/v1/detections/analyze` - Analyze video
- `GET /api/v1/detections/history` - Detection history
- `GET /api/v1/detections/{id}` - Get detection details

### Camera Endpoints
- `GET /api/v1/cameras` - List cameras
- `POST /api/v1/cameras` - Add camera
- `GET /api/v1/cameras/{id}/status` - Camera status
- `POST /api/v1/cameras/{id}/start` - Start camera

### Alert Endpoints
- `GET /api/v1/alerts` - List alerts
- `POST /api/v1/alerts` - Create alert
- `PUT /api/v1/alerts/{id}/acknowledge` - Acknowledge alert

### Evidence Endpoints
- `GET /api/v1/evidence` - List evidence
- `GET /api/v1/evidence/{id}` - Get evidence
- `POST /api/v1/evidence/collect` - Collect evidence

### Mapping Endpoints
- `GET /api/v1/mapping/zones` - List zones
- `GET /api/v1/mapping/incidents` - Incident locations
- `POST /api/v1/mapping/threat` - Report threat location

## 🚨 Production Deployment

### Using Gunicorn

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Setup

```bash
# Production environment
export API_DEBUG=False
export LOG_LEVEL=WARNING
export REDIS_URL=redis://your-redis-server:6379
```

## 🔍 Monitoring

### Health Checks
- `GET /health` - Basic health check
- `GET /api/v1/system/status` - Detailed system status
- `GET /api/v1/cameras/status` - Camera status overview

### Logging
- Application logs: `data/logs/`
- Error tracking: Integrated with loguru
- Performance metrics: Built-in timing

### Metrics
- Detection accuracy
- Processing latency
- Camera uptime
- System resource usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` endpoint
- **Issues**: Report bugs and feature requests
- **Discord**: Join our development server
- **Email**: Contact the development team

---

**Built with ❤️ for HackPSU 2026**
