import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""

    # Flask config
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    # YOLO config
    YOLO_MODEL = os.getenv('YOLO_MODEL', 'yolov8n.pt')  # Options: yolov8n, yolov8s, yolov8m, yolov8l
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.35'))

    # Detection config
    JAM_LEVEL_LOW = int(os.getenv('JAM_LEVEL_LOW', '5'))
    JAM_LEVEL_MEDIUM = int(os.getenv('JAM_LEVEL_MEDIUM', '10'))
    JAM_LEVEL_HIGH = int(os.getenv('JAM_LEVEL_HIGH', '20'))
