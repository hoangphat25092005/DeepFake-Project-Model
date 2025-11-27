import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from api.models.database import db, InferenceResult
from api.services.inference_service import detector
import time

inference_bp = Blueprint('inference', __name__, url_prefix='/api/inference')


def allowed_file(filename, allowed_extensions):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@inference_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict if uploaded image is fake or real
    
    Request:
        - file: Image file (multipart/form-data)
        - threshold: Optional threshold (default 0.5)
    
    Response:
        {
            "success": bool,
            "result": {
                "prediction": float,
                "is_fake": bool,
                "confidence": float,
                "processing_time": float
            },
            "inference_id": int
        }
    """
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
    if not allowed_file(file.filename, allowed_extensions):
        return jsonify({
            'success': False,
            'error': f'File type not allowed. Allowed types: {", ".join(allowed_extensions)}'
        }), 400
    
    try:
        # Get threshold from request
        threshold = float(request.form.get('threshold', 0.5))
        
        # Read image bytes
        image_bytes = file.read()
        
        # Perform inference
        result = detector.predict_image(image_bytes, threshold=threshold)
        
        # Save to database
        inference_result = InferenceResult(
            image_name=secure_filename(file.filename),
            prediction=result['prediction'],
            is_fake=result['is_fake'],
            confidence=result['confidence'],
            threshold=threshold,
            processing_time=result['processing_time']
        )
        
        db.session.add(inference_result)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'result': result,
            'inference_id': inference_result.id
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@inference_bp.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple images
    
    Request:
        - files: Multiple image files
        - threshold: Optional threshold
    
    Response:
        {
            "success": bool,
            "results": [...]
        }
    """
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if len(files) == 0:
        return jsonify({'success': False, 'error': 'No files selected'}), 400
    
    try:
        threshold = float(request.form.get('threshold', 0.5))
        results = []
        inference_ids = []
        
        for file in files:
            if file and allowed_file(file.filename, {'png', 'jpg', 'jpeg', 'bmp'}):
                image_bytes = file.read()
                result = detector.predict_image(image_bytes, threshold=threshold)
                
                # Save to database
                inference_result = InferenceResult(
                    image_name=secure_filename(file.filename),
                    prediction=result['prediction'],
                    is_fake=result['is_fake'],
                    confidence=result['confidence'],
                    threshold=threshold,
                    processing_time=result['processing_time']
                )
                
                db.session.add(inference_result)
                db.session.flush()
                
                results.append({
                    'filename': file.filename,
                    'result': result,
                    'inference_id': inference_result.id
                })
                inference_ids.append(inference_result.id)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@inference_bp.route('/history', methods=['GET'])
def get_history():
    """
    Get inference history
    
    Query params:
        - limit: Max number of results (default 50)
        - offset: Offset for pagination (default 0)
    
    Response:
        {
            "success": bool,
            "count": int,
            "results": [...]
        }
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        results = InferenceResult.query.order_by(
            InferenceResult.created_at.desc()
        ).limit(limit).offset(offset).all()
        
        total_count = InferenceResult.query.count()
        
        return jsonify({
            'success': True,
            'count': len(results),
            'total': total_count,
            'results': [r.to_dict() for r in results]
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@inference_bp.route('/result/<int:inference_id>', methods=['GET'])
def get_result(inference_id):
    """
    Get specific inference result by ID
    
    Response:
        {
            "success": bool,
            "result": {...}
        }
    """
    try:
        result = InferenceResult.query.get(inference_id)
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'Result not found'
            }), 404
        
        return jsonify({
            'success': True,
            'result': result.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@inference_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Get statistics about inferences
    
    Response:
        {
            "success": bool,
            "stats": {
                "total_inferences": int,
                "total_fake": int,
                "total_real": int,
                "fake_percentage": float,
                "avg_processing_time": float
            }
        }
    """
    try:
        total = InferenceResult.query.count()
        total_fake = InferenceResult.query.filter_by(is_fake=True).count()
        total_real = InferenceResult.query.filter_by(is_fake=False).count()
        
        avg_time = db.session.query(
            db.func.avg(InferenceResult.processing_time)
        ).scalar()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_inferences': total,
                'total_fake': total_fake,
                'total_real': total_real,
                'fake_percentage': (total_fake / total * 100) if total > 0 else 0,
                'avg_processing_time': float(avg_time) if avg_time else 0
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
