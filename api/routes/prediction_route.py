"""
Deepfake prediction routes
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
import traceback
from datetime import datetime

prediction_bp = Blueprint('prediction', __name__)

# Will be set by app.py
model_loader = None
minio_handler = None
db_service = None
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}


def init_prediction_routes(model_service, minio_service, database_service=None):
    """Initialize prediction routes with services"""
    global model_loader, minio_handler, db_service
    model_loader = model_service
    minio_handler = minio_service
    db_service = database_service


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    """
    Predict if a single image is real or fake
    ---
    tags:
      - Prediction
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: Image file to analyze (JPEG, PNG, etc.)
    responses:
      200:
        description: Prediction successful
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            filename:
              type: string
              example: test_image.jpg
            timestamp:
              type: string
              example: "2024-12-09T12:00:00.123456"
            prediction:
              type: object
              properties:
                label:
                  type: string
                  example: FAKE
                is_fake:
                  type: boolean
                  example: true
                confidence:
                  type: number
                  example: 0.8756
                scores:
                  type: object
                  properties:
                    real:
                      type: number
                      example: 0.1244
                    fake:
                      type: number
                      example: 0.8756
            image_info:
              type: object
              properties:
                width:
                  type: integer
                  example: 512
                height:
                  type: integer
                  example: 512
                size_bytes:
                  type: integer
                  example: 45678
            result_image_url:
              type: string
              example: "http://localhost:9000/deepfake-results/result_20241209_120000_test.jpg?..."
            result_filename:
              type: string
              example: result_20241209_120000_test.jpg
            note:
              type: string
              example: Save result_image_url to database for later retrieval
      400:
        description: Bad request (no image, invalid format, etc.)
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: false
            error:
              type: string
              example: No image file provided
      500:
        description: Server error
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: false
            error:
              type: string
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Prepare filename
        original_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_filename = f"result_{timestamp}_{original_filename}"
        
        print(f"\n[PREDICT] Processing: {original_filename}")
        
        # Run prediction
        print("  Running D³ model...")
        prediction = model_loader.predict(image)
        
        print(f"  Result: {prediction['label']} ({prediction['confidence']:.2%})")
        
        # Upload result image to MinIO
        print("  Uploading result to MinIO...")
        minio_url = minio_handler.upload_result_image(
            image=image,
            result_filename=result_filename,
            prediction_data=prediction
        )
        
        # Save to MongoDB database
        if db_service:
            print("  Saving to MongoDB...")
            try:
                db_record = db_service.save_prediction(
                    filename=original_filename,
                    minio_url=minio_url,
                    prediction_data={
                        'label': prediction['label'],
                        'is_fake': prediction['is_fake'],
                        'confidence': prediction['confidence'],
                        'fake_score': prediction['fake_score'],
                        'real_score': prediction['real_score']
                    },
                    image_info={
                        'width': image.width,
                        'height': image.height,
                        'format': image.format,
                        'size_bytes': len(image_bytes)
                    }
                )
                print(f"  ✅ Saved to database with ID: {db_record.id}")
            except Exception as db_error:
                print(f"  ⚠️  Database save failed: {db_error}")
                # Continue anyway - don't fail the request
        
        print(f"✅ Complete!\n")
        
        # Prepare response
        response = {
            'success': True,
            'filename': original_filename,
            'timestamp': datetime.now().isoformat(),
            
            # Prediction results
            'prediction': {
                'label': prediction['label'],
                'is_fake': prediction['is_fake'],
                'confidence': round(prediction['confidence'], 4),
                'scores': {
                    'real': round(prediction['real_score'], 4),
                    'fake': round(prediction['fake_score'], 4)
                }
            },
            
            # Image info
            'image_info': {
                'width': image.width,
                'height': image.height,
                'size_bytes': len(image_bytes)
            },
            
            # MinIO storage URL (save this in database)
            'result_image_url': minio_url,
            'result_filename': result_filename,
            
            # Note for database
            'note': 'Save result_image_url to database for later retrieval'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Predict multiple images in batch
    ---
    tags:
      - Prediction
    consumes:
      - multipart/form-data
    parameters:
      - name: images
        in: formData
        type: file
        required: true
        description: Multiple image files (max 50)
    responses:
      200:
        description: Batch prediction successful
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            total_images:
              type: integer
              example: 3
            successful:
              type: integer
              example: 3
            failed:
              type: integer
              example: 0
            results:
              type: array
              items:
                type: object
                properties:
                  index:
                    type: integer
                  filename:
                    type: string
                  prediction:
                    type: object
                  result_image_url:
                    type: string
                  result_filename:
                    type: string
            errors:
              type: array
              items:
                type: object
      400:
        description: Bad request
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: false
            error:
              type: string
      500:
        description: Server error
    """
    try:
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
        files = request.files.getlist('images')
        
        if len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
        if len(files) > 50:
            return jsonify({
                'success': False,
                'error': 'Maximum 50 images per batch'
            }), 400
        
        results = []
        errors = []
        
        print(f"\n[BATCH] Processing {len(files)} images...")
        
        for idx, file in enumerate(files):
            try:
                if not allowed_file(file.filename):
                    errors.append({
                        'index': idx,
                        'filename': file.filename,
                        'error': 'File type not allowed'
                    })
                    continue
                
                # Read image
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Prepare filename
                original_filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_filename = f"result_{timestamp}_{idx}_{original_filename}"
                
                print(f"  [{idx+1}/{len(files)}] {original_filename}")
                
                # Run prediction
                prediction = model_loader.predict(image)
                
                # Upload result to MinIO
                minio_url = minio_handler.upload_result_image(
                    image=image,
                    result_filename=result_filename,
                    prediction_data=prediction
                )
                
                # Save to MongoDB database
                if db_service:
                    try:
                        db_service.save_prediction(
                            filename=original_filename,
                            minio_url=minio_url,
                            prediction_data={
                                'label': prediction['label'],
                                'is_fake': prediction['is_fake'],
                                'confidence': prediction['confidence'],
                                'fake_score': prediction['fake_score'],
                                'real_score': prediction['real_score']
                            },
                            image_info={
                                'width': image.width,
                                'height': image.height,
                                'format': image.format,
                                'size_bytes': len(image_bytes)
                            }
                        )
                    except Exception as db_error:
                        print(f"    ⚠️  Database save failed: {db_error}")
                
                results.append({
                    'index': idx,
                    'filename': original_filename,
                    'prediction': {
                        'label': prediction['label'],
                        'is_fake': prediction['is_fake'],
                        'confidence': round(prediction['confidence'], 4),
                        'scores': {
                            'real': round(prediction['real_score'], 4),
                            'fake': round(prediction['fake_score'], 4)
                        }
                    },
                    'result_image_url': minio_url,
                    'result_filename': result_filename
                })
                
            except Exception as e:
                errors.append({
                    'index': idx,
                    'filename': file.filename,
                    'error': str(e)
                })
        
        print(f" Batch complete: {len(results)} succeeded, {len(errors)} failed\n")
        
        return jsonify({
            'success': True,
            'total_images': len(files),
            'successful': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors if errors else None
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    

