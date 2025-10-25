"""
System-wide constants for Violence Detection Backend
"""

# API Constants
API_PREFIX = "/api/v1"
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
REQUEST_TIMEOUT = 30  # seconds

# Video Processing Constants
DEFAULT_FPS = 30
DEFAULT_RESOLUTION = (640, 480)
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# ML Model Constants
VIOLENCE_CLASSES = ['non_violence', 'violence']
MULTI_CLASS_CATEGORIES = [
    'normal',
    'fighting',
    'assault',
    'vandalism',
    'theft',
    'harassment',
    'intimidation',
    'weapon_use',
    'property_damage',
    'public_disturbance',
    'trespassing',
    'gang_activity',
    'other_violence'
]

# Detection Constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
HIGH_CONFIDENCE_THRESHOLD = 0.9
LOW_CONFIDENCE_THRESHOLD = 0.5

# Camera Constants
MAX_CAMERAS = 10
CAMERA_TIMEOUT = 30  # seconds
STREAM_BUFFER_SIZE = 10  # frames
RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY = 5  # seconds

# Evidence Collection Constants
EVIDENCE_RETENTION_DAYS = 30
MAX_EVIDENCE_SIZE = 500 * 1024 * 1024  # 500MB
EVIDENCE_FORMATS = ['.mp4', '.jpg', '.json']

# Notification Constants
ALERT_PRIORITY_LOW = "low"
ALERT_PRIORITY_MEDIUM = "medium"
ALERT_PRIORITY_HIGH = "high"
ALERT_PRIORITY_CRITICAL = "critical"

# Redis Keys
REDIS_KEYS = {
    'CAMERA_STATUS': 'camera:status',
    'DETECTION_RESULTS': 'detection:results',
    'INCIDENT_TRACKER': 'incident:tracker',
    'EVIDENCE_QUEUE': 'evidence:queue',
    'ALERT_QUEUE': 'alert:queue',
    'SYSTEM_STATUS': 'system:status'
}

# HTTP Status Codes
HTTP_STATUS = {
    'OK': 200,
    'CREATED': 201,
    'BAD_REQUEST': 400,
    'UNAUTHORIZED': 401,
    'FORBIDDEN': 403,
    'NOT_FOUND': 404,
    'CONFLICT': 409,
    'UNPROCESSABLE_ENTITY': 422,
    'INTERNAL_SERVER_ERROR': 500,
    'SERVICE_UNAVAILABLE': 503
}

# Error Messages
ERROR_MESSAGES = {
    'INVALID_VIDEO_FORMAT': 'Invalid video format. Supported formats: mp4, avi, mov, mkv, webm',
    'VIDEO_TOO_LARGE': 'Video file too large. Maximum size: 100MB',
    'CAMERA_NOT_FOUND': 'Camera not found',
    'CAMERA_OFFLINE': 'Camera is offline',
    'MODEL_NOT_LOADED': 'ML model not loaded',
    'INVALID_CONFIDENCE': 'Confidence threshold must be between 0.0 and 1.0',
    'REDIS_CONNECTION_FAILED': 'Failed to connect to Redis',
    'EVIDENCE_NOT_FOUND': 'Evidence not found',
    'INCIDENT_NOT_FOUND': 'Incident not found',
    'INVALID_COORDINATES': 'Invalid coordinates provided',
    'MAPPING_ERROR': 'Building map error',
    'NOTIFICATION_FAILED': 'Failed to send notification'
}

# Success Messages
SUCCESS_MESSAGES = {
    'DETECTION_COMPLETE': 'Detection completed successfully',
    'EVIDENCE_SAVED': 'Evidence saved successfully',
    'ALERT_SENT': 'Alert sent successfully',
    'CAMERA_CONNECTED': 'Camera connected successfully',
    'MODEL_LOADED': 'Model loaded successfully',
    'INCIDENT_CREATED': 'Incident created successfully',
    'NOTIFICATION_SENT': 'Notification sent successfully'
}

# File Extensions
FILE_EXTENSIONS = {
    'VIDEO': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
    'IMAGE': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    'AUDIO': ['.mp3', '.wav', '.aac', '.flac'],
    'DOCUMENT': ['.pdf', '.doc', '.docx', '.txt'],
    'DATA': ['.json', '.csv', '.xml', '.yaml', '.yml']
}

# Time Constants
TIME_FORMATS = {
    'ISO': '%Y-%m-%dT%H:%M:%S.%fZ',
    'DATETIME': '%Y-%m-%d %H:%M:%S',
    'DATE': '%Y-%m-%d',
    'TIME': '%H:%M:%S'
}

# Building Zones
BUILDING_ZONES = {
    'ENTRANCE': 'entrance',
    'LOBBY': 'lobby',
    'HALLWAY': 'hallway',
    'CLASSROOM': 'classroom',
    'OFFICE': 'office',
    'CAFETERIA': 'cafeteria',
    'GYMNASIUM': 'gymnasium',
    'PARKING': 'parking',
    'OUTDOOR': 'outdoor',
    'OTHER': 'other'
}

# Threat Levels
THREAT_LEVELS = {
    'LOW': 1,
    'MEDIUM': 2,
    'HIGH': 3,
    'CRITICAL': 4
}

# System Status
SYSTEM_STATUS = {
    'ONLINE': 'online',
    'OFFLINE': 'offline',
    'MAINTENANCE': 'maintenance',
    'ERROR': 'error'
}

# Camera Status
CAMERA_STATUS = {
    'ONLINE': 'online',
    'OFFLINE': 'offline',
    'CONNECTING': 'connecting',
    'ERROR': 'error',
    'MAINTENANCE': 'maintenance'
}

# Detection Status
DETECTION_STATUS = {
    'PENDING': 'pending',
    'PROCESSING': 'processing',
    'COMPLETED': 'completed',
    'FAILED': 'failed',
    'CANCELLED': 'cancelled'
}
