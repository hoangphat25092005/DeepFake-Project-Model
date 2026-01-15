from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
import os

def check_minio_connection(endpoint, access_key, secret_key, secure=False):
    """
    Checks for a connection to a MinIO server.
    """
    try:
        # Initialize MinIO client
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        # Attempt a basic operation, like listing buckets, to verify connectivity
        client.list_buckets()
        print(f"Successfully connected to MinIO server at {endpoint}")
        return True
    except S3Error as e:
        print(f"MinIO connection error: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False



if __name__ == "__main__":
    load_dotenv()
    # Replace with your MinIO server details
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY")
    minio_secure = os.getenv("MINIO_SECURE", "False").lower() == "true"

    if check_minio_connection(minio_endpoint, minio_access_key, minio_secret_key, minio_secure):
        print("MinIO appears to be installed and running.")
    else:
        print("MinIO might not be installed or accessible, or connection details are incorrect.")