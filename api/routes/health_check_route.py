from flask import Blueprint, jsonify

health_bp = Blueprint('health', __name__, url_prefix='/api/health')

minio_handler = None  # will be later set by app.py

def init_health_routes(minio_service):
    global minio_handler
    minio_handler = minio_service


@health_bp.route('/', methods=['GET'])
def home():
    try:
        if minio_handler:
            minio_handler.client.list_buckets()

        return jsonify({
            'status': 'healthy',
            'model': 'loaded',
            'minio': 'connected',
            'bucket': minio_handler.bucket_name
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500
    
