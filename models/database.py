"""
Database models for D³ Deepfake Detection API
Stores prediction results with MinIO URLs
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class PredictionRecord(Base):
    """
    Stores prediction results with MinIO URL
    """
    __tablename__ = 'predictions'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # File information
    filename = Column(String(255), nullable=False, index=True)
    minio_url = Column(Text, nullable=False)  # Presigned URL from MinIO
    
    # Prediction results
    label = Column(String(10), nullable=False, index=True)  # REAL or FAKE
    is_fake = Column(Boolean, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    fake_score = Column(Float, nullable=False)
    real_score = Column(Float, nullable=False)
    
    # Image metadata
    image_width = Column(Integer)
    image_height = Column(Integer)
    image_format = Column(String(10))  # PNG, JPEG, etc.
    file_size = Column(Integer)  # Size in bytes
    
    # Model information
    model_name = Column(String(100), default='D3_CLIP_ViT-L/14')
    model_version = Column(String(50), default='v1.0')
    
    # Optional tracking
    user_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<PredictionRecord(id={self.id}, filename='{self.filename}', label='{self.label}', confidence={self.confidence:.2%})>"
    
    def to_dict(self, include_minio_url=True):
        """
        Convert record to dictionary
        
        Args:
            include_minio_url: Include MinIO URL (may be expired)
            
        Returns:
            Dictionary with all record data
        """
        data = {
            'id': self.id,
            'filename': self.filename,
            
            # Prediction results
            'prediction': {
                'label': self.label,
                'is_fake': self.is_fake,
                'confidence': round(self.confidence, 4),
                'scores': {
                    'real': round(self.real_score, 4),
                    'fake': round(self.fake_score, 4)
                }
            },
            
            # Image information
            'image_info': {
                'width': self.image_width,
                'height': self.image_height,
                'format': self.image_format,
                'size_bytes': self.file_size
            },
            
            # Model information
            'model': {
                'name': self.model_name,
                'version': self.model_version
            },
            
            # Timestamps
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            
            # Optional tracking
            'user_id': self.user_id,
            'session_id': self.session_id
        }
        
        # Include MinIO URL if requested (it may expire after 7 days)
        if include_minio_url:
            data['minio_url'] = self.minio_url
            data['minio_note'] = 'URL expires after 7 days. Use /refresh-url endpoint to regenerate.'
        
        return data
    
    def to_dict_compact(self):
        """
        Compact version without MinIO URL (for listing)
        """
        return {
            'id': self.id,
            'filename': self.filename,
            'label': self.label,
            'is_fake': self.is_fake,
            'confidence': round(self.confidence, 4),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class BatchPrediction(Base):
    """
    Optional: Track batch prediction jobs
    """
    __tablename__ = 'batch_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Statistics
    total_images = Column(Integer, nullable=False)
    successful = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    
    # Status
    status = Column(String(20), default='processing')  # processing, completed, failed
    
    # Optional tracking
    user_id = Column(String(100), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<BatchPrediction(id={self.id}, batch_id='{self.batch_id}', total={self.total_images}, status='{self.status}')>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'statistics': {
                'total_images': self.total_images,
                'successful': self.successful,
                'failed': self.failed,
                'success_rate': round((self.successful / self.total_images * 100), 2) if self.total_images > 0 else 0
            },
            'status': self.status,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class ModelMetrics(Base):
    """
    Optional: Track model performance metrics over time
    """
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Metrics
    total_predictions = Column(Integer, default=0)
    fake_predictions = Column(Integer, default=0)
    real_predictions = Column(Integer, default=0)
    
    average_confidence = Column(Float, default=0.0)
    average_fake_confidence = Column(Float, default=0.0)
    average_real_confidence = Column(Float, default=0.0)
    
    # Time period
    date = Column(DateTime, default=datetime.utcnow, unique=True, index=True)
    
    def __repr__(self):
        return f"<ModelMetrics(date={self.date.date()}, total={self.total_predictions})>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'date': self.date.date().isoformat() if self.date else None,
            'predictions': {
                'total': self.total_predictions,
                'fake': self.fake_predictions,
                'real': self.real_predictions
            },
            'average_confidence': {
                'overall': round(self.average_confidence, 4),
                'fake': round(self.average_fake_confidence, 4),
                'real': round(self.average_real_confidence, 4)
            }
        }


# Create indexes for better query performance
from sqlalchemy import Index

# Composite indexes for common queries
Index('idx_predictions_label_created', PredictionRecord.label, PredictionRecord.created_at.desc())
Index('idx_predictions_is_fake_created', PredictionRecord.is_fake, PredictionRecord.created_at.desc())
Index('idx_predictions_user_created', PredictionRecord.user_id, PredictionRecord.created_at.desc())
Index('idx_predictions_session_created', PredictionRecord.session_id, PredictionRecord.created_at.desc())