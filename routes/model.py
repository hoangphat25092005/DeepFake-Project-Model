from flask import Blueprint, jsonify, request
from api.services.minio_service import minio_client
from api.services.inference_service import detector

model_bp = Blueprint('model', __name__, url_prefix='/api/model')


@model_bp.route('/info', methods=['GET'])
def get_model_info():
    """
    Get information about the current model
    
    Response:
        {
            "success": bool,
            "info": {
                "loaded": bool,
                "model_name": str,
                "architecture": str,
                "device": str
            }
        }
    """
    try:
        return jsonify({
            'success': True,
            'info': {
                'loaded': detector.loaded,
                'model_name': detector.config.MODEL_NAME if detector.config else None,
                'architecture': detector.config.MODEL_ARCH if detector.config else None,
                'device': detector.device if detector.device else None
            }
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@model_bp.route('/load', methods=['POST'])
def load_model():
    """
    Load model into memory
    
    Response:
        {
            "success": bool,
            "message": str
        }
    """
    try:
        detector.load_model()
        return jsonify({
            'success': True,
            'message': 'Model loaded successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@model_bp.route('/unload', methods=['POST'])
def unload_model():
    """
    Unload model from memory
    
    Response:
        {
            "success": bool,
            "message": str
        }
    """
    try:
        detector.unload_model()
        return jsonify({
            'success': True,
            'message': 'Model unloaded successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@model_bp.route('/list', methods=['GET'])
def list_models():
    """
    List all models in MinIO
    
    Response:
        {
            "success": bool,
            "models": [...]
        }
    """
    try:
        models = minio_client.list_models()
        return jsonify({
            'success': True,
            'count': len(models),
            'models': models
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@model_bp.route('/upload', methods=['POST'])
def upload_model():
    """
    Upload a new model to MinIO
    
    Request:
        - file: Model checkpoint file
        - name: Optional custom name
    
    Response:
        {
            "success": bool,
            "model_name": str
        }
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        # Save temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Upload to MinIO
        object_name = request.form.get('name', file.filename)
        minio_client.upload_model(tmp_path, object_name)
        
        # Clean up
        import os
        os.remove(tmp_path)
        
        return jsonify({
            'success': True,
            'message': 'Model uploaded successfully',
            'model_name': object_name
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
