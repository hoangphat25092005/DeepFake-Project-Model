"""
Database service for storing prediction results with MinIO URLs
MongoDB version
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from bson import ObjectId
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

from api.models.database import PredictionRecord, BatchPrediction, ModelMetrics


class DatabaseService:
    """Handle MongoDB database operations for predictions"""
    
    def __init__(self, database_url=None):
        """Initialize MongoDB connection"""
        self.database_url = database_url or os.getenv('MONGODB_URL', 'mongodb://localhost:27017/')
        self.database_name = os.getenv('MONGODB_DATABASE', 'd3_deepfake')
        
        print(f"[Database] Connecting to MongoDB...")
        
        try:
            # Create MongoDB client
            self.client = MongoClient(
                self.database_url,
                maxPoolSize=int(os.getenv('DATABASE_POOL_SIZE', 50)),
                serverSelectionTimeoutMS=5000
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database
            self.db = self.client[self.database_name]
            
            # Get collections
            self.predictions = self.db['predictions']
            self.batch_predictions_collection = self.db['batch_predictions']
            self.model_metrics_collection = self.db['model_metrics']
            
            # Create indexes
            self._create_indexes()
            
            print(f"  ✅ MongoDB connected successfully to database: {self.database_name}")
            
        except ConnectionFailure as e:
            print(f"  ❌ Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create MongoDB indexes for better query performance"""
        try:
            # Predictions collection indexes
            self.predictions.create_index([('filename', ASCENDING)])
            self.predictions.create_index([('label', ASCENDING)])
            self.predictions.create_index([('is_fake', ASCENDING)])
            self.predictions.create_index([('created_at', DESCENDING)])
            self.predictions.create_index([('user_id', ASCENDING)])
            self.predictions.create_index([('session_id', ASCENDING)])
            
            # Compound indexes for common queries
            self.predictions.create_index([('label', ASCENDING), ('created_at', DESCENDING)])
            self.predictions.create_index([('is_fake', ASCENDING), ('created_at', DESCENDING)])
            self.predictions.create_index([('user_id', ASCENDING), ('created_at', DESCENDING)])
            
            # Batch predictions indexes
            self.batch_predictions_collection.create_index([('batch_id', ASCENDING)], unique=True)
            self.batch_predictions_collection.create_index([('status', ASCENDING)])
            self.batch_predictions_collection.create_index([('created_at', DESCENDING)])
            
            # Model metrics indexes
            self.model_metrics_collection.create_index([('date', DESCENDING)], unique=True)
            
            print(f"  ✅ MongoDB indexes created successfully")
        except Exception as e:
            print(f"  ⚠️  Warning: Some indexes may already exist: {e}")
    
    def save_prediction(self, filename, minio_url, prediction_data, image_info, 
                       user_id=None, session_id=None):
        """
        Save prediction with MinIO URL to MongoDB
        
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
        try:
            record = PredictionRecord(
                filename=filename,
                minio_url=minio_url,
                label=prediction_data.get('label'),
                is_fake=prediction_data.get('is_fake'),
                confidence=prediction_data.get('confidence'),
                fake_score=prediction_data.get('fake_score'),
                real_score=prediction_data.get('real_score'),
                image_width=image_info.get('width'),
                image_height=image_info.get('height'),
                image_format=image_info.get('format'),
                file_size=image_info.get('size_bytes'),
                model_name='D3_CLIP_ViT-L/14',
                model_version='v1.0',
                user_id=user_id,
                session_id=session_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Insert into MongoDB
            result = self.predictions.insert_one(record.to_mongo_dict())
            record._id = result.inserted_id
            
            print(f"  ✅ Saved to MongoDB: ID={record.id}, MinIO URL stored")
            
            return record
            
        except Exception as e:
            print(f"  ❌ Error saving prediction: {e}")
            raise
    
    def get_prediction_by_id(self, prediction_id):
        """Get prediction by ID"""
        try:
            # Convert string ID to ObjectId
            if isinstance(prediction_id, str):
                prediction_id = ObjectId(prediction_id)
            
            doc = self.predictions.find_one({'_id': prediction_id})
            if doc:
                record = PredictionRecord.from_mongo_dict(doc)
                return record.to_dict()
            return None
            
        except Exception as e:
            print(f"  ❌ Error retrieving prediction: {e}")
            return None
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        try:
            cursor = self.predictions.find().sort('created_at', DESCENDING).limit(limit)
            records = [PredictionRecord.from_mongo_dict(doc) for doc in cursor]
            return [r.to_dict() for r in records]
            
        except Exception as e:
            print(f"  ❌ Error retrieving predictions: {e}")
            return []
    
    def get_predictions_by_label(self, label, limit=10):
        """Get predictions filtered by label (REAL or FAKE)"""
        try:
            cursor = self.predictions.find(
                {'label': label.upper()}
            ).sort('created_at', DESCENDING).limit(limit)
            
            records = [PredictionRecord.from_mongo_dict(doc) for doc in cursor]
            return [r.to_dict() for r in records]
            
        except Exception as e:
            print(f"  ❌ Error retrieving predictions by label: {e}")
            return []
    
    def get_statistics(self):
        """Get overall statistics using MongoDB aggregation"""
        try:
            # Total count
            total = self.predictions.count_documents({})
            fake_count = self.predictions.count_documents({'is_fake': True})
            real_count = self.predictions.count_documents({'is_fake': False})
            
            # Average confidence for fake predictions
            fake_pipeline = [
                {'$match': {'is_fake': True}},
                {'$group': {'_id': None, 'avg_confidence': {'$avg': '$confidence'}}}
            ]
            fake_result = list(self.predictions.aggregate(fake_pipeline))
            avg_fake_conf = fake_result[0]['avg_confidence'] if fake_result else 0
            
            # Average confidence for real predictions
            real_pipeline = [
                {'$match': {'is_fake': False}},
                {'$group': {'_id': None, 'avg_confidence': {'$avg': '$confidence'}}}
            ]
            real_result = list(self.predictions.aggregate(real_pipeline))
            avg_real_conf = real_result[0]['avg_confidence'] if real_result else 0
            
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
            
        except Exception as e:
            print(f"  ❌ Error calculating statistics: {e}")
            return {
                'total_predictions': 0,
                'fake_count': 0,
                'real_count': 0,
                'fake_percentage': 0,
                'real_percentage': 0,
                'average_confidence': {'fake': 0, 'real': 0}
            }
    
    def regenerate_minio_url(self, prediction_id, minio_handler):
        """Regenerate expired MinIO URL and update MongoDB"""
        try:
            # Convert string ID to ObjectId
            if isinstance(prediction_id, str):
                prediction_id = ObjectId(prediction_id)
            
            doc = self.predictions.find_one({'_id': prediction_id})
            if not doc:
                return None
            
            record = PredictionRecord.from_mongo_dict(doc)
            
            # Extract object name from old URL or use filename
            import re
            match = re.search(r'/([^/]+/[^?]+)', record.minio_url)
            if match:
                object_name = match.group(1).split('/', 1)[1]
            else:
                object_name = f"results/{record.filename}"
            
            # Generate new presigned URL
            new_url = minio_handler.get_file_url(object_name)
            
            # Update in MongoDB
            self.predictions.update_one(
                {'_id': prediction_id},
                {
                    '$set': {
                        'minio_url': new_url,
                        'updated_at': datetime.utcnow()
                    }
                }
            )
            
            record.minio_url = new_url
            record.updated_at = datetime.utcnow()
            
            return record.to_dict()
            
        except Exception as e:
            print(f"  ❌ Error regenerating MinIO URL: {e}")
            return None
    
    def search_predictions(self, query, limit=10):
        """Search predictions by filename using regex"""
        try:
            # Use case-insensitive regex search
            import re
            pattern = re.compile(query, re.IGNORECASE)
            
            cursor = self.predictions.find(
                {'filename': {'$regex': pattern}}
            ).sort('created_at', DESCENDING).limit(limit)
            
            records = [PredictionRecord.from_mongo_dict(doc) for doc in cursor]
            return [r.to_dict() for r in records]
            
        except Exception as e:
            print(f"  ❌ Error searching predictions: {e}")
            return []
    
    def close(self):
        """Close MongoDB connections"""
        try:
            self.client.close()
            print("  ✅ MongoDB connections closed")
        except Exception as e:
            print(f"  ⚠️  Warning closing connections: {e}")
    
    # Additional helper methods
    
    def get_predictions_by_user(self, user_id, limit=10):
        """Get predictions for a specific user"""
        try:
            cursor = self.predictions.find(
                {'user_id': user_id}
            ).sort('created_at', DESCENDING).limit(limit)
            
            records = [PredictionRecord.from_mongo_dict(doc) for doc in cursor]
            return [r.to_dict() for r in records]
            
        except Exception as e:
            print(f"  ❌ Error retrieving user predictions: {e}")
            return []
    
    def get_predictions_by_session(self, session_id, limit=10):
        """Get predictions for a specific session"""
        try:
            cursor = self.predictions.find(
                {'session_id': session_id}
            ).sort('created_at', DESCENDING).limit(limit)
            
            records = [PredictionRecord.from_mongo_dict(doc) for doc in cursor]
            return [r.to_dict() for r in records]
            
        except Exception as e:
            print(f"  ❌ Error retrieving session predictions: {e}")
            return []
    
    def delete_prediction(self, prediction_id):
        """Delete a prediction by ID"""
        try:
            if isinstance(prediction_id, str):
                prediction_id = ObjectId(prediction_id)
            
            result = self.predictions.delete_one({'_id': prediction_id})
            return result.deleted_count > 0
            
        except Exception as e:
            print(f"  ❌ Error deleting prediction: {e}")
            return False
    
    def get_predictions_count(self, filters=None):
        """Get count of predictions with optional filters"""
        try:
            if filters is None:
                filters = {}
            return self.predictions.count_documents(filters)
        except Exception as e:
            print(f"  ❌ Error counting predictions: {e}")
            return 0
