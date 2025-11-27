from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class InferenceResult(db.Model):
    __tablename__ = 'detection_deepfake'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, foreign_key='users.id', nullable=True)  
    file_path = db.Column(db.String(255), nullable=True)  # Original filename
    prediction = db.Column(db.Float, nullable=False)  # Raw prediction score
    result = db.Column(db.String(50), nullable=True)  # 'fake' or 'real'
    result_url = db.Column(db.String(500), nullable=True)  # URL to from the MinIO storage
    model_version = db.Column(db.String(100), nullable=True)  
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Additional fields 
    confidence = db.Column(db.Float, nullable=True)
    threshold = db.Column(db.Float, default=0.5)
    processing_time = db.Column(db.Float, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'file_path': self.file_path,
            'prediction': self.prediction,
            'result': self.result,
            'result_url': self.result_url,
            'model_version': self.model_version,
            'confidence': self.confidence,
            'threshold': self.threshold,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }