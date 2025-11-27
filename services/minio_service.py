from minio import Minio
from minio.error import S3Error
import io
from typing import Optional
from datetime import timedelta
from PIL import Image

class MinioService:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        self.client = Minio(
            endpoint, 
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        self.bucket_name = 'd3-results'
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            raise Exception(f"Error ensuring bucket exists: {e}")
        
    def upload_image(self, image: Image.Image, object_name: str, content_type: str = "image/jpeg") -> str:
        try:
            self.client.put_object(
                self.bucket_name,
                object_name,
                io.BytesIO(image.tobytes()),
                length=len(image.tobytes()),
                content_type=content_type
            )
        
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=timedelta(days=7)
            )
            return url

        except S3Error as e:
            raise Exception(f"Error uploading image: {e}")

    def upload_file(self, file_data: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
        try:
            self.client.put_object(
                self.bucket_name,
                object_name,
                io.BytesIO(file_data),
                length=len(file_data),
                content_type=content_type
            )
        
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=timedelta(days=7)
            )
            return url

        except S3Error as e:
            raise Exception(f"Error uploading file: {e}")
        
    def get_url(self, object_name: str) -> Optional[str]:
        try:
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=timedelta(days=7)
            )
            return url
        except S3Error as e:
            raise Exception(f"Error generating URL: {e}")
        
    def delete_object(self, object_name: str) -> bool:
        try:
            self.client.remove_object(self.bucket_name, object_name)
            return True
        except S3Error as e:
            raise Exception(f"Error deleting object: {e}")
        
    def list_objects(self, prefix: str = '') -> list:
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=prefix)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            raise Exception(f"Error listing objects: {e}")