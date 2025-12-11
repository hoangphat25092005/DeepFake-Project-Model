from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class PredictionRecord(Base):
    """Table to store prediction results"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=True)
    minio_url = Column(String(500), nullable=True)
    
    # Prediction results
    label = Column(String(10), nullable=False)
    is_fake = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    fake_score = Column(Float, nullable=False)
    real_score = Column(Float, nullable=False)
    
    # Image metadata
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    image_format = Column(String(50), nullable=True)
    file_size = Column(Integer, nullable=True)
    
    # Model info
    model_name = Column(String(100), default='D3_CLIP_ViT-L/14')
    model_version = Column(String(50), default='v1.0')
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Optional
    user_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'prediction': {
                'label': self.label,
                'is_fake': self.is_fake,
                'confidence': self.confidence,
                'scores': {
                    'fake': self.fake_score,
                    'real': self.real_score
                }
            },
            'image_info': {
                'width': self.image_width,
                'height': self.image_height,
                'format': self.image_format,
                'size_bytes': self.file_size
            },
            'timestamps': {
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None
            },
            'storage': {
                'local_path': self.file_path,
                'minio_url': self.minio_url
            }
        }


class DatabaseManager:
    """Manage database connections"""
    
    def __init__(self, database_url=None):
        if database_url is None:
            database_url = os.getenv('DATABASE_URL')
        
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        print("Database tables created")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def add_prediction(self, prediction_data):
        """Add prediction record"""
        session = self.get_session()
        try:
            record = PredictionRecord(**prediction_data)
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_all_predictions(self, limit=100, offset=0):
        """Get all predictions"""
        session = self.get_session()
        try:
            return session.query(PredictionRecord).order_by(
                PredictionRecord.created_at.desc()
            ).limit(limit).offset(offset).all()
        finally:
            session.close()
    
    def get_statistics(self):
        """Get statistics"""
        session = self.get_session()
        try:
            total = session.query(PredictionRecord).count()
            fake_count = session.query(PredictionRecord).filter_by(is_fake=True).count()
            real_count = total - fake_count
            
            return {
                'total_predictions': total,
                'fake_predictions': fake_count,
                'real_predictions': real_count,
                'fake_percentage': (fake_count / total * 100) if total > 0 else 0
            }
        finally:
            session.close()