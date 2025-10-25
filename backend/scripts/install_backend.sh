#!/bin/bash
# install_backend.sh - Violence Detection Backend Installation Script

echo "🚀 Installing Violence Detection Backend..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3 is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 is not installed. Please install pip."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    if [ -f "../Requirements.txt" ]; then
        pip install -r ../Requirements.txt
        print_success "Dependencies installed"
    else
        print_error "Requirements.txt not found"
        exit 1
    fi
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    # Create main directories
    mkdir -p {core,models,camera,coordination,evidence,mapping,api/routes,utils,notifications,tests,scripts,data/{analysis,logs,temp,cache}}
    
    # Create __init__.py files
    find . -type d -name "backend" -prune -o -type d -exec touch {}/__init__.py \;
    
    print_success "Directory structure created"
}

# Set permissions
set_permissions() {
    print_status "Setting file permissions..."
    chmod +x scripts/*.py 2>/dev/null || true
    chmod +x scripts/*.sh 2>/dev/null || true
    print_success "Permissions set"
}

# Create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Violence Detection Backend Environment Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379

# Database Configuration
DATABASE_URL=sqlite:///./violence_detection.db

# Paths
DATA_DIR=./data
EVIDENCE_DIR=./evidence
MODEL_DIR=./models

# Model Paths
CNN_MODEL_PATH=./models/violence_model_best.pth
MULTI_CLASS_MODEL_PATH=./models/multi_class_violence_model_best.pth

# Processing Configuration
MAX_FRAMES_PER_VIDEO=30
CLIP_DURATION_BEFORE=10
CLIP_DURATION_AFTER=10
CONFIDENCE_THRESHOLD=0.7
VIOLENCE_THRESHOLD=0.8

# Camera Configuration
MAX_CAMERAS=10
CAMERA_TIMEOUT=30
STREAM_FPS=30

# Notification Settings (configure these)
DISCORD_WEBHOOK_URL=
EMAIL_SMTP_SERVER=
EMAIL_FROM=
EMAIL_PASSWORD=
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=

# Security
API_KEY=
JWT_SECRET=

# Logging
LOG_LEVEL=INFO
EOF
        print_success "Environment file created"
    else
        print_warning "Environment file already exists"
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Python imports
    python3 -c "
import sys
sys.path.append('.')
try:
    from config import settings
    print('✅ Config module imported successfully')
except ImportError as e:
    print(f'❌ Config import failed: {e}')
    sys.exit(1)

try:
    from utils.logger import setup_logger
    print('✅ Logger module imported successfully')
except ImportError as e:
    print(f'❌ Logger import failed: {e}')
    sys.exit(1)

print('✅ All core modules imported successfully')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Main installation process
main() {
    echo "🎯 Violence Detection Backend Installation"
    echo "=========================================="
    
    # Change to backend directory
    cd "$(dirname "$0")/.."
    
    # Run installation steps
    check_python
    check_pip
    create_venv
    activate_venv
    install_dependencies
    create_directories
    set_permissions
    create_env_file
    test_installation
    
    echo ""
    print_success "🎉 Backend installation complete!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Configure your .env file with your settings"
    echo "2. Install Redis: brew install redis (macOS) or apt-get install redis (Ubuntu)"
    echo "3. Start Redis: redis-server"
    echo "4. Run the backend: python main.py"
    echo "5. Access the API docs: http://localhost:8000/docs"
    echo ""
    echo "🔧 Development commands:"
    echo "  Activate venv: source venv/bin/activate"
    echo "  Run backend: python main.py"
    echo "  Run tests: python -m pytest tests/"
    echo "  Install deps: pip install -r ../Requirements.txt"
    echo ""
}

# Run main function
main "$@"
