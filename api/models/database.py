"""
Database models for D³ Deepfake Detection API
MongoDB version - Stores prediction results with MinIO URLs
"""

from datetime import datetime
from typing import Optional, Dict, Any
from bson import ObjectId


class PredictionRecord:
    """
    Stores prediction results with MinIO URL in MongoDB
    """
    
    def __init__(self, **kwargs):
        """Initialize a prediction record"""
        self._id = kwargs.get('_id', None)
        self.filename = kwargs.get('filename')
        self.minio_url = kwargs.get('minio_url')
        
        # Prediction results
        self.label = kwargs.get('label')
        self.is_fake = kwargs.get('is_fake')
        self.confidence = kwargs.get('confidence')
        self.fake_score = kwargs.get('fake_score')
        self.real_score = kwargs.get('real_score')
        
        # Image metadata
        self.image_width = kwargs.get('image_width')
        self.image_height = kwargs.get('image_height')
        self.image_format = kwargs.get('image_format')
        self.file_size = kwargs.get('file_size')
        
        # Model information
        self.model_name = kwargs.get('model_name', 'D3_CLIP_ViT-L/14')
        self.model_version = kwargs.get('model_version', 'v1.0')
        
        # Optional tracking
        self.user_id = kwargs.get('user_id')
        self.session_id = kwargs.get('session_id')
        
        # Timestamps
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
    
    @property
    def id(self):
        """Get string representation of MongoDB ObjectId"""
        return str(self._id) if self._id else None
    
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
    
    def to_mongo_dict(self):
        """Convert to MongoDB document format"""
        doc = {
            'filename': self.filename,
            'minio_url': self.minio_url,
            'label': self.label,
            'is_fake': self.is_fake,
            'confidence': self.confidence,
            'fake_score': self.fake_score,
            'real_score': self.real_score,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'image_format': self.image_format,
            'file_size': self.file_size,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        if self._id:
            doc['_id'] = self._id
        return doc
    
    @staticmethod
    def from_mongo_dict(doc):
        """Create PredictionRecord from MongoDB document"""
        if not doc:
            return None
        return PredictionRecord(**doc)


class BatchPrediction:
    """
    Track batch prediction jobs in MongoDB
    """
    
    def __init__(self, **kwargs):
        self._id = kwargs.get('_id', None)
        self.batch_id = kwargs.get('batch_id')
        self.total_images = kwargs.get('total_images', 0)
        self.successful = kwargs.get('successful', 0)
        self.failed = kwargs.get('failed', 0)
        self.status = kwargs.get('status', 'processing')
        self.user_id = kwargs.get('user_id')
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.completed_at = kwargs.get('completed_at')
    
    @property
    def id(self):
        return str(self._id) if self._id else None
    
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
    
    def to_mongo_dict(self):
        """Convert to MongoDB document format"""
        doc = {
            'batch_id': self.batch_id,
            'total_images': self.total_images,
            'successful': self.successful,
            'failed': self.failed,
            'status': self.status,
            'user_id': self.user_id,
            'created_at': self.created_at,
            'completed_at': self.completed_at
        }
        if self._id:
            doc['_id'] = self._id
        return doc
    
    @staticmethod
    def from_mongo_dict(doc):
        """Create BatchPrediction from MongoDB document"""
        if not doc:
            return None
        return BatchPrediction(**doc)


class ModelMetrics:
    """
    Track model performance metrics over time in MongoDB
    """
    
    def __init__(self, **kwargs):
        self._id = kwargs.get('_id', None)
        self.total_predictions = kwargs.get('total_predictions', 0)
        self.fake_predictions = kwargs.get('fake_predictions', 0)
        self.real_predictions = kwargs.get('real_predictions', 0)
        self.average_confidence = kwargs.get('average_confidence', 0.0)
        self.average_fake_confidence = kwargs.get('average_fake_confidence', 0.0)
        self.average_real_confidence = kwargs.get('average_real_confidence', 0.0)
        self.date = kwargs.get('date', datetime.utcnow())
    
    @property
    def id(self):
        return str(self._id) if self._id else None
    
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
    
    def to_mongo_dict(self):
        """Convert to MongoDB document format"""
        doc = {
            'total_predictions': self.total_predictions,
            'fake_predictions': self.fake_predictions,
            'real_predictions': self.real_predictions,
            'average_confidence': self.average_confidence,
            'average_fake_confidence': self.average_fake_confidence,
            'average_real_confidence': self.average_real_confidence,
            'date': self.date
        }
        if self._id:
            doc['_id'] = self._id
        return doc
    
    @staticmethod
    def from_mongo_dict(doc):
        """Create ModelMetrics from MongoDB document"""
        if not doc:
            return None
        return ModelMetrics(**doc)