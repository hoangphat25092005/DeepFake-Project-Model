"""
Database service for storing prediction results with MinIO URLs
"""

from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

from api.models.database import Base, PredictionRecord


class DatabaseService:
    """Handle database operations for predictions"""
    
    def __init__(self, database_url=None):
        """Initialize database connection"""
        self.database_url = database_url or os.getenv('DATABASE_URL')
        
        if not self.database_url:
            raise ValueError("DATABASE_URL is not configured")
        
        print(f"[Database] Connecting to PostgreSQL...")
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=int(os.getenv('DATABASE_POOL_SIZE', 5)),
            max_overflow=int(os.getenv('DATABASE_MAX_OVERFLOW', 10)),
            pool_pre_ping=True,
            echo=False
        )
        
        # Create session factory
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.SessionFactory)
        
        # Create tables
        self._create_tables()
        
        print(f"  ✅ Database connected successfully")
    
    def _create_tables(self):
        """Create database tables"""
        try:
            Base.metadata.create_all(self.engine)
            print(f"  ✅ Database tables verified/created")
        except Exception as e:
            print(f"  ❌ Error creating tables: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def save_prediction(self, filename, minio_url, prediction_data, image_info, 
                       user_id=None, session_id=None):
        """
        Save prediction with MinIO URL to database
        
        Args:
            filename: Original filename
            minio_url: MinIO presigned URL
            prediction_data: Dict with label, is_fake, confidence, scores
            image_info: Dict with width, height, format, size_bytes
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            PredictionRecord object
        """
        with self.get_session() as session:
            record = PredictionRecord(
                # File information
                filename=filename,
                minio_url=minio_url,
                
                # Prediction results
                label=prediction_data.get('label'),
                is_fake=prediction_data.get('is_fake'),
                confidence=prediction_data.get('confidence'),
                fake_score=prediction_data.get('fake_score'),
                real_score=prediction_data.get('real_score'),
                
                # Image metadata
                image_width=image_info.get('width'),
                image_height=image_info.get('height'),
                image_format=image_info.get('format'),
                file_size=image_info.get('size_bytes'),
                
                # Model info
                model_name='D3_CLIP_ViT-L/14',
                model_version='v1.0',
                
                # Optional
                user_id=user_id,
                session_id=session_id
            )
            
            session.add(record)
            session.flush()
            
            print(f"  ✅ Saved to database: ID={record.id}, MinIO URL stored")
            
            return record
    
    def get_prediction_by_id(self, prediction_id):
        """Get prediction by ID"""
        with self.get_session() as session:
            record = session.query(PredictionRecord).filter_by(id=prediction_id).first()
            if record:
                return record.to_dict()
            return None
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        with self.get_session() as session:
            records = session.query(PredictionRecord).order_by(
                desc(PredictionRecord.created_at)
            ).limit(limit).all()
            
            return [r.to_dict() for r in records]
    
    def get_predictions_by_label(self, label, limit=10):
        """Get predictions filtered by label (REAL or FAKE)"""
        with self.get_session() as session:
            records = session.query(PredictionRecord).filter_by(
                label=label.upper()
            ).order_by(desc(PredictionRecord.created_at)).limit(limit).all()
            
            return [r.to_dict() for r in records]
    
    def get_statistics(self):
        """Get overall statistics"""
        with self.get_session() as session:
            total = session.query(PredictionRecord).count()
            fake_count = session.query(PredictionRecord).filter_by(is_fake=True).count()
            real_count = session.query(PredictionRecord).filter_by(is_fake=False).count()
            
            # Average confidence by label
            avg_fake_conf = session.query(func.avg(PredictionRecord.confidence)).filter_by(
                is_fake=True
            ).scalar() or 0
            
            avg_real_conf = session.query(func.avg(PredictionRecord.confidence)).filter_by(
                is_fake=False
            ).scalar() or 0
            
            return {
                'total_predictions': total,
                'fake_count': fake_count,
                'real_count': real_count,
                'fake_percentage': (fake_count / total * 100) if total > 0 else 0,
                'real_percentage': (real_count / total * 100) if total > 0 else 0,
                'average_confidence': {
                    'fake': round(avg_fake_conf, 4),
                    'real': round(avg_real_conf, 4)
                }
            }
    
    def regenerate_minio_url(self, prediction_id, minio_handler):
        """Regenerate expired MinIO URL and update database"""
        with self.get_session() as session:
            record = session.query(PredictionRecord).filter_by(id=prediction_id).first()
            if not record:
                return None
            
            # Extract object name from old URL or use filename
            # Assuming MinIO URL format: http://host:port/bucket/object?params
            import re
            match = re.search(r'/([^/]+/[^?]+)', record.minio_url)
            if match:
                object_name = match.group(1).split('/', 1)[1]
            else:
                object_name = f"results/{record.filename}"
            
            # Generate new presigned URL
            new_url = minio_handler.get_file_url(object_name)
            record.minio_url = new_url
            
            return record.to_dict()
    
    def search_predictions(self, query, limit=10):
        """Search predictions by filename"""
        with self.get_session() as session:
            records = session.query(PredictionRecord).filter(
                PredictionRecord.filename.ilike(f'%{query}%')
            ).order_by(desc(PredictionRecord.created_at)).limit(limit).all()
            
            return [r.to_dict() for r in records]
    
    def close(self):
        """Close database connections"""
        self.Session.remove()
        self.engine.dispose()
        print("  ✅ Database connections closed")