"""
Utility functions for the API
"""
import os
import hashlib
from datetime import datetime


def generate_api_key():
    """Generate a random API key"""
    return hashlib.sha256(os.urandom(32)).hexdigest()


def allowed_file(filename, allowed_extensions):
    """Check if filename has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def format_bytes(size):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def get_file_hash(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
