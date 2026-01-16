"""
Database service for storing prediction results with MinIO URLs
PostgreSQL version
"""

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


class DatabaseService:
    """Handle PostgreSQL database operations for predictions"""
    
    def __init__(self, database_url=None):
        """Initialize PostgreSQL connection pool"""
        if database_url is None:
            # Build connection string from environment variables
            self.db_host = os.getenv('POSTGRES_HOST', 'localhost')
            self.db_port = os.getenv('POSTGRES_PORT', '5432')
            self.db_name = os.getenv('POSTGRES_DB', 'deepfake_db')
            self.db_user = os.getenv('POSTGRES_USER', 'postgres')
            self.db_password = os.getenv('POSTGRES_PASSWORD', 'postgres')
            
            database_url = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        
        print(f"[Database] Connecting to PostgreSQL...")
        
        try:
            # Create connection pool
            self.pool = SimpleConnectionPool(
                1,
                int(os.getenv('DATABASE_POOL_SIZE', 5)),
                database_url
            )
            
            # Test connection
            conn = self.pool.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            self.pool.putconn(conn)
            
            print(f"  ✅ PostgreSQL connected successfully to database: {self.db_name}")
            
        except psycopg2.Error as e:
            print(f"  ❌ Failed to connect to PostgreSQL: {e}")
            raise
    
    def _get_connection(self):
        """Get a connection from the pool"""
        return self.pool.getconn()
    
    def _put_connection(self, conn):
        """Return a connection to the pool"""
        self.pool.putconn(conn)
    
    def save_prediction(self, filename, minio_url, prediction_data, image_info, 
                       user_id=None, session_id=None):
        """
        Save prediction with MinIO URL to PostgreSQL
        
        Args:
            filename: Original filename
            minio_url: MinIO presigned URL
            prediction_data: Dict with label, is_fake, confidence, scores
            image_info: Dict with width, height, format, size_bytes
            user_id: Optional user identifier
            session_id: Optional session identifier (not used in PostgreSQL schema currently)
            
        Returns:
            Dict with saved prediction record
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Insert prediction
            query = """
            INSERT INTO predictions 
            (filename, prediction, confidence, is_fake, result_image_url, 
             image_width, image_height, image_size_bytes, user_id, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            RETURNING id, filename, prediction, confidence, is_fake, result_image_url, 
                      image_width, image_height, image_size_bytes, user_id, created_at, updated_at
            """
            
            cursor.execute(query, (
                filename,
                prediction_data.get('label'),
                prediction_data.get('confidence'),
                prediction_data.get('is_fake'),
                minio_url,
                image_info.get('width'),
                image_info.get('height'),
                image_info.get('size_bytes'),
                user_id
            ))
            
            result = cursor.fetchone()
            conn.commit()
            
            print(f"  ✅ Saved to PostgreSQL: ID={result['id']}, MinIO URL stored")
            
            return dict(result)
            
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            print(f"  ❌ Error saving prediction: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)
    
    def get_prediction_by_id(self, prediction_id):
        """Get prediction by ID"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = "SELECT * FROM predictions WHERE id = %s"
            cursor.execute(query, (prediction_id,))
            result = cursor.fetchone()
            
            return dict(result) if result else None
            
        except psycopg2.Error as e:
            print(f"  ❌ Error retrieving prediction: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Limit to reasonable max
            limit = min(limit, 100)
            
            query = """
            SELECT * FROM predictions 
            ORDER BY created_at DESC 
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            print(f"  ❌ Error retrieving predictions: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)
    
    def get_predictions_by_label(self, label, limit=10):
        """Get predictions filtered by label (REAL or FAKE)"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Limit to reasonable max
            limit = min(limit, 100)
            label = label.upper()
            
            query = """
            SELECT * FROM predictions 
            WHERE prediction = %s
            ORDER BY created_at DESC 
            LIMIT %s
            """
            cursor.execute(query, (label, limit))
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            print(f"  ❌ Error retrieving predictions by label: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)
    
    def get_statistics(self):
        """Get overall statistics using PostgreSQL view"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Query the prediction_stats view
            query = "SELECT * FROM prediction_stats"
            cursor.execute(query)
            result = cursor.fetchone()
            
            if result:
                total = result['total_predictions'] or 0
                fake_count = result['fake_count'] or 0
                real_count = result['real_count'] or 0
                
                return {
                    'total_predictions': total,
                    'fake_count': fake_count,
                    'real_count': real_count,
                    'fake_percentage': (fake_count / total * 100) if total > 0 else 0,
                    'real_percentage': (real_count / total * 100) if total > 0 else 0,
                    'average_confidence': float(result['avg_confidence'] or 0),
                    'last_prediction': result['last_prediction'].isoformat() if result['last_prediction'] else None
                }
            else:
                return {
                    'total_predictions': 0,
                    'fake_count': 0,
                    'real_count': 0,
                    'fake_percentage': 0,
                    'real_percentage': 0,
                    'average_confidence': 0,
                    'last_prediction': None
                }
            
        except psycopg2.Error as e:
            print(f"  ❌ Error calculating statistics: {e}")
            return {
                'total_predictions': 0,
                'fake_count': 0,
                'real_count': 0,
                'fake_percentage': 0,
                'real_percentage': 0,
                'average_confidence': 0,
                'last_prediction': None
            }
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)
    
    def regenerate_minio_url(self, prediction_id, minio_handler):
        """Regenerate expired MinIO URL and update PostgreSQL"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get prediction
            query = "SELECT * FROM predictions WHERE id = %s"
            cursor.execute(query, (prediction_id,))
            result = cursor.fetchone()
            
            if not result:
                return None
            
            # Extract object name from old URL or use filename
            import re
            match = re.search(r'/([^/]+/[^?]+)', result['result_image_url'])
            if match:
                object_name = match.group(1).split('/', 1)[1]
            else:
                object_name = f"results/{result['filename']}"
            
            # Generate new presigned URL
            new_url = minio_handler.get_file_url(object_name)
            
            # Update in PostgreSQL
            update_query = """
            UPDATE predictions 
            SET result_image_url = %s, updated_at = NOW()
            WHERE id = %s
            RETURNING *
            """
            cursor.execute(update_query, (new_url, prediction_id))
            updated = cursor.fetchone()
            conn.commit()
            
            return dict(updated)
            
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            print(f"  ❌ Error regenerating MinIO URL: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)
    
    def search_predictions(self, query_str, limit=10):
        """Search predictions by filename using LIKE"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Limit to reasonable max
            limit = min(limit, 100)
            
            search_pattern = f"%{query_str}%"
            query = """
            SELECT * FROM predictions 
            WHERE filename ILIKE %s
            ORDER BY created_at DESC 
            LIMIT %s
            """
            cursor.execute(query, (search_pattern, limit))
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            print(f"  ❌ Error searching predictions: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)
    
    def close(self):
        """Close PostgreSQL connections"""
        try:
            self.pool.closeall()
            print("  ✅ PostgreSQL connections closed")
        except Exception as e:
            print(f"  ⚠️  Warning closing connections: {e}")
    
    # Additional helper methods
    
    def get_predictions_by_user(self, user_id, limit=10):
        """Get predictions for a specific user"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Limit to reasonable max
            limit = min(limit, 100)
            
            query = """
            SELECT * FROM predictions 
            WHERE user_id = %s
            ORDER BY created_at DESC 
            LIMIT %s
            """
            cursor.execute(query, (user_id, limit))
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            print(f"  ❌ Error retrieving user predictions: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)
    
    def get_predictions_by_session(self, session_id, limit=10):
        """Get predictions for a specific session - not used in current schema"""
        # PostgreSQL schema doesn't have session_id, return empty
        return []
    
    def delete_prediction(self, prediction_id):
        """Delete a prediction by ID"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = "DELETE FROM predictions WHERE id = %s"
            cursor.execute(query, (prediction_id,))
            conn.commit()
            
            return cursor.rowcount > 0
            
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            print(f"  ❌ Error deleting prediction: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)
    
    def get_predictions_count(self, filters=None):
        """Get count of predictions with optional filters"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if filters is None:
                query = "SELECT COUNT(*) FROM predictions"
                cursor.execute(query)
            else:
                # Build WHERE clause from filters
                where_parts = []
                params = []
                for key, value in filters.items():
                    where_parts.append(f"{key} = %s")
                    params.append(value)
                
                where_clause = " AND ".join(where_parts)
                query = f"SELECT COUNT(*) FROM predictions WHERE {where_clause}"
                cursor.execute(query, params)
            
            result = cursor.fetchone()
            return result[0] if result else 0
            
        except psycopg2.Error as e:
            print(f"  ❌ Error counting predictions: {e}")
            return 0
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._put_connection(conn)
