import os
from flask import Flask, jsonify
from flask_cors import CORS
from api.config.config import config
from api.models.database import db
from api.services.minio_service import minio_client
from api.services.inference_service import detector
from api.routes.inference import inference_bp
from api.routes.model import model_bp


def create_app(config_name='default'):
    """Application factory"""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize CORS
    CORS(app)
    
    # Initialize database
    db.init_app(app)
    
    # Initialize MinIO
    with app.app_context():
        minio_client.initialize(app.config)
        detector.initialize(app.config)
    
    # Register blueprints
    app.register_blueprint(inference_bp)
    app.register_blueprint(model_bp)
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'D3 Deepfake Detection API',
            'model_loaded': detector.loaded
        }), 200
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        return jsonify({
            'service': 'D3 Deepfake Detection API',
            'version': '1.0.0',
            'endpoints': {
                'health': '/health',
                'inference': {
                    'predict': 'POST /api/inference/predict',
                    'predict_batch': 'POST /api/inference/predict-batch',
                    'history': 'GET /api/inference/history',
                    'result': 'GET /api/inference/result/<id>',
                    'stats': 'GET /api/inference/stats'
                },
                'model': {
                    'info': 'GET /api/model/info',
                    'load': 'POST /api/model/load',
                    'unload': 'POST /api/model/unload',
                    'list': 'GET /api/model/list',
                    'upload': 'POST /api/model/upload'
                }
            }
        }), 200
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'success': False, 'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    return app


def init_db(app):
    """Initialize database tables"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully")


if __name__ == '__main__':
    app = create_app(os.getenv('FLASK_ENV', 'development'))
    
    # Create database tables
    init_db(app)
    
    # Run application
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=app.config['DEBUG']
    )
