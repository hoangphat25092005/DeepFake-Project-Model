import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from api.models.database import db, InferenceResult
from api.services.inference_service import detector
from api.services.minio_service import MinioService  # ⬅
from datetime import datetime
import uuid

inference_bp = Blueprint('inference', __name__, url_prefix='/api/inference')

# Initialize MinIO service (configure in app.py)
minio_service = None

def init_minio_service(minio):
    global minio_service
    minio_service = minio


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
        - user_id: Optional user ID
        - store_image: Optional bool to store image in MinIO (default true)
    
    Response:
        {
            "success": bool,
            "result": {
                "prediction": float,
                "is_fake": bool,
                "confidence": float,
                "processing_time": float,
                "result_url": str  # 
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
        # Get parameters from request
        threshold = float(request.form.get('threshold', 0.5))
        user_id = request.form.get('user_id', None)  
        store_image = request.form.get('store_image', 'true').lower() == 'true' 
        
        # Read image bytes
        image_bytes = file.read()
        
        # Perform inference
        result = detector.predict_image(image_bytes, threshold=threshold)
        
        # Upload to MinIO if requested
        result_url = None
        if store_image and minio_service:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            file_extension = secure_filename(file.filename).rsplit('.', 1)[-1]
            object_name = f"inferences/{timestamp}_{unique_id}.{file_extension}"
            
            # Upload to MinIO
            result_url = minio_service.upload_image(
                image_bytes,
                object_name,
                content_type=f"image/{file_extension}"
            )
        
        # Save to database
        inference_result = InferenceResult(
            user_id=int(user_id) if user_id else None,  
            file_path=secure_filename(file.filename),
            prediction=result['prediction'],
            result='fake' if result['is_fake'] else 'real',  
            result_url=result_url,  
            model_version=detector.get_model_version(),  
            confidence=result['confidence'],
            threshold=threshold,
            processing_time=result['processing_time']
        )
        
        db.session.add(inference_result)
        db.session.commit()
        
        # Add result_url to response
        response_result = result.copy()
        response_result['result_url'] = result_url
        
        return jsonify({
            'success': True,
            'result': response_result,
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
        - user_id: Optional user ID
        - store_images: Optional bool to store images in MinIO
    
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
        user_id = request.form.get('user_id', None)  
        store_images = request.form.get('store_images', 'true').lower() == 'true'  
        results = []
        
        for idx, file in enumerate(files):
            if file and allowed_file(file.filename, {'png', 'jpg', 'jpeg', 'bmp'}):
                image_bytes = file.read()
                result = detector.predict_image(image_bytes, threshold=threshold)
                
                # Upload to MinIO if requested
                result_url = None
                if store_images and minio_service:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    unique_id = str(uuid.uuid4())[:8]
                    file_extension = secure_filename(file.filename).rsplit('.', 1)[-1]
                    object_name = f"inferences/{timestamp}_{unique_id}_{idx}.{file_extension}"
                    
                    result_url = minio_service.upload_image(
                        image_bytes,
                        object_name,
                        content_type=f"image/{file_extension}"
                    )
                
                # Save to database
                inference_result = InferenceResult(
                    user_id=int(user_id) if user_id else None,  
                    file_path=secure_filename(file.filename),
                    prediction=result['prediction'],
                    result='fake' if result['is_fake'] else 'real',
                    result_url=result_url,  
                    model_version=detector.get_model_version(), 
                    confidence=result['confidence'],
                    threshold=threshold,
                    processing_time=result['processing_time']
                )
                
                db.session.add(inference_result)
                db.session.flush()
                
                response_result = result.copy()
                response_result['result_url'] = result_url
                
                results.append({
                    'filename': file.filename,
                    'result': response_result,
                    'inference_id': inference_result.id
                })
        
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


# ...existing code for other routes (history, stats, etc.)...