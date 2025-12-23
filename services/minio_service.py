from minio import Minio
from minio.error import S3Error
import os
from datetime import timedelta
from dotenv import load_dotenv
import io
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

class MinioHandler:
    """Handle MinIO storage for prediction result images"""
    
    def __init__(self):
        self.endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        self.access_key = os.getenv('MINIO_ACCESS_KEY')
        self.secret_key = os.getenv('MINIO_SECRET_KEY')
        self.secure = os.getenv('MINIO_SECURE', 'False').lower() == 'true'
        self.bucket_name = os.getenv('MINIO_BUCKET', 'deepfake-results')
        
        print(f"[MinIO] Initializing...")
        print(f"  Endpoint: {self.endpoint}")
        print(f"  Bucket: {self.bucket_name}")
        
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if not exists"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"Created bucket: {self.bucket_name}")
            else:
                print(f"Bucket exists: {self.bucket_name}")
        except S3Error as e:
            raise Exception(f"Error ensuring bucket exists: {e}")
    
    def upload_result_image(self, image, result_filename, prediction_data):
        """
        Upload result image with prediction overlay to MinIO
        
        Args:
            image: PIL Image object (original image)
            result_filename: Filename for storage (e.g., "20241209_120000_test.jpg")
            prediction_data: Dict with prediction results
            
        Returns:
            str: Presigned URL to access the result image (valid for 7 days)
        """
        try:
            # Create result image with prediction overlay
            result_image = self._create_result_image(image, prediction_data)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            result_image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            # Upload to MinIO
            file_size = img_byte_arr.getbuffer().nbytes
            
            self.client.put_object(
                self.bucket_name,
                result_filename,
                img_byte_arr,
                file_size,
                content_type='image/jpeg'
            )
            
            # Generate presigned URL (valid for 7 days)
            url = self.client.presigned_get_object(
                self.bucket_name,
                result_filename,
                expires=timedelta(days=7)
            )
            
            print(f"Uploaded result: {result_filename}")
            return url
            
        except S3Error as e:
            print(f"Upload error: {e}")
            raise Exception("Failed to upload result image to MinIO")
    
    def _create_result_image(self, image, prediction_data):
        """
        Create result image with prediction overlay
        
        Args:
            image: PIL Image
            prediction_data: Dict with 'label', 'confidence', 'is_fake'
            
        Returns:
            PIL Image with overlay
        """
        # Create a copy to avoid modifying original
        result_img = image.copy()
        draw = ImageDraw.Draw(result_img)
        
        # Get prediction info
        label = prediction_data.get('label', 'UNKNOWN')
        confidence = prediction_data.get('confidence', 0.0)
        is_fake = prediction_data.get('is_fake', False)
        
        # Colors
        color = (255, 0, 0) if is_fake else (0, 255, 0)  # Red for FAKE, Green for REAL
        
        # Text
        text = f"{label}: {confidence:.2%}"
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        # Calculate text size and position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position at top center
        x = (result_img.width - text_width) // 2
        y = 20
        
        # Draw background rectangle
        padding = 10
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 128)
        )
        
        # Draw text
        draw.text((x, y), text, fill=color, font=font)
        
        return result_img
    
    def upload_simple_image(self, file_object, object_name, content_type='image/jpeg'):
        """
        Upload simple file to MinIO (without overlay)
        
        Args:
            file_object: File-like object
            object_name: Filename in MinIO
            content_type: MIME type
            
        Returns:
            str: Presigned URL
        """
        try:
            file_object.seek(0, 2)
            file_size = file_object.tell()
            file_object.seek(0)
            
            self.client.put_object(
                self.bucket_name,
                object_name,
                file_object,
                file_size,
                content_type=content_type
            )
            
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=timedelta(days=7)
            )
            
            return url
            
        except S3Error as e:
            print(f"Upload error: {e}")
            raise

    def upload_video(self, file_path, object_name, content_type='video/mp4'):
        """
        Upload a video file to MinIO.

        Args:
            file_path: Path to the video file on disk
            object_name: Filename to use in MinIO
            content_type: MIME type (default: video/mp4)

        Returns:
            str: Presigned URL to access the video (valid for 7 days)
        """
        try:
            with open(file_path, 'rb') as f:
                file_size = os.path.getsize(file_path)
                self.client.put_object(
                    self.bucket_name,
                    object_name,
                    f,
                    file_size,
                    content_type=content_type
                )
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=timedelta(days=7)
            )
            print(f"Uploaded video: {object_name}")
            return url
        except S3Error as e:
            print(f"Upload error: {e}")
            raise Exception("Failed to upload video to MinIO")
        
    def upload_frame_results_json(self, frame_predictions, result_filename):
        """Upload frame results as JSON to MinIO"""
        import json
        try:
            data = io.BytesIO(json.dumps(frame_predictions).encode('utf-8'))
            file_size = data.getbuffer().nbytes

            self.client.put_object(
                self.bucket_name,
                result_filename,
                data,
                file_size,
                content_type='application/json'
            )

            url = self.client.presigned_get_object(
                self.bucket_name,
                result_filename,
                expires=timedelta(days=7)
            )
            print(f"Uploaded frame results JSON: {result_filename}")
            return url
        except S3Error as e:
            print(f"Upload error: {e}")
            raise Exception("Failed to upload frame results JSON to MinIO")
    
    def get_file_url(self, object_name, expires_days=7):
        """Get presigned URL for existing file"""
        try:
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=timedelta(days=expires_days)
            )
            return url
        except S3Error as e:
            print(f"URL generation error: {e}")
            return None
    
    def delete_file(self, object_name):
        """Delete file from MinIO"""
        try:
            self.client.remove_object(self.bucket_name, object_name)
            print(f"Deleted: {object_name}")
            return True
        except S3Error as e:
            print(f"Delete error: {e}")
            return False
    
    def list_files(self, prefix=None):
        """List all files in bucket"""
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=prefix)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            print(f"List error: {e}")
            return []