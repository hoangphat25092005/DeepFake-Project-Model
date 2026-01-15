import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # MongoDB Configuration
    MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'd3_deepfake')
    MONGODB_USERNAME = os.getenv('MONGODB_USERNAME', '')
    MONGODB_PASSWORD = os.getenv('MONGODB_PASSWORD', '')
    
    # MongoDB connection string with authentication (if provided)
    if MONGODB_USERNAME and MONGODB_PASSWORD:
        MONGODB_URL = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_URL.replace('mongodb://', '')}"
    
    # MinIO Configuration
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
    MINIO_SECURE = os.getenv('MINIO_SECURE', 'False').lower() == 'true'
    MINIO_BUCKET_NAME = os.getenv('MINIO_BUCKET_NAME', 'd3-models')
    
    # Model Configuration
    MODEL_NAME = 'classifier.pth'
    MODEL_ARCH = 'CLIP:ViT-L/14'
    GRANULARITY = 14
    
    # Upload Configuration
    UPLOAD_FOLDER = 'api/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    
    # Inference Configuration
    BATCH_SIZE = 1
    USE_FP16 = True
    DEVICE = os.getenv('CUDA_DEVICE', '1')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
