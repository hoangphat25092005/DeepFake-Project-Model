"""
D³ Deepfake Detection API
Main application entry point
"""

from flask import Flask, jsonify
from flasgger import Swagger
import os
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, '/mnt/mmlab2024nas/danh/phatlh/D3')

# Import services
from api.utils.model_loader import D3ModelLoader
from api.services.minio_service import MinioHandler
from api.services.db_service_mongodb import DatabaseService

# Import routes
from api.routes import health_bp, prediction_bp, video_prediction_bp
from api.routes.health_check_route import init_health_routes
from api.routes.prediction_route import init_prediction_routes
from api.routes.video_prediction_route import init_video_routes

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))

# Swagger configuration
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "D³ Deepfake Detection API",
        "description": "API for detecting deepfake images using D³ (Diverse Deepfake Detection) model with CLIP architecture",
        "version": "1.0.0",
        "contact": {
            "name": "API Support",
            "email": "support@example.com"
        }
    },
    "basePath": "/",
    "schemes": ["http"],
    "consumes": ["multipart/form-data"],
    "produces": ["application/json"],
    "tags": [
        {
            "name": "Health",
            "description": "Health check endpoints"
        },
        {
            "name": "Prediction",
            "description": "Deepfake detection endpoints"
        }, 
        {
            "name": "Video Prediction",
            "description": "Video deepfake detection endpoints"
        }, 
        {
            "name": "Video Upload",
            "description": "Video upload endpoints"
        }
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# Get configuration from environment
CHECKPOINT_PATH = os.getenv('MODEL_CHECKPOINT_PATH')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize services
print("\n" + "="*70)
print("D³ DEEPFAKE DETECTION API")
print("="*70)

try:
    print("\n[1/3] Loading D³ model...")
    model_loader = D3ModelLoader(checkpoint_path=CHECKPOINT_PATH)
    
    print("\n[2/3] Connecting to MinIO...")
    minio_handler = MinioHandler()
    
    print("\n[3/3] Connecting to MongoDB...")
    db_service = DatabaseService()
    
    print("\n" + "="*70)
    print("API READY!")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\nINITIALIZATION FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Initialize routes with services
init_health_routes(minio_handler)
init_prediction_routes(model_loader, minio_handler, db_service)
init_video_routes(model_loader, minio_handler)

# Register blueprints
app.register_blueprint(health_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(video_prediction_bp)

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum file size is 100MB.'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Single image prediction',
            '/batch_predict': 'POST - Batch prediction',
            '/docs': 'GET - API documentation'
        }
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Get configuration
    port = int(os.getenv('FLASK_PORT', 5000))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Print startup info
    print(f"\nStarting API server on {host}:{port}")
    print(f"API Documentation: http://localhost:{port}/docs")
    print(f"Home: http://localhost:{port}/")
    print(f"Health: http://localhost:{port}/health\n")
    
    # Run app
    app.run(
        debug=debug,
        host=host,
        port=port,
        threaded=True
    )