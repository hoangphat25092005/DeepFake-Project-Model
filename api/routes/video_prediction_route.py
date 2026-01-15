from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from pathlib import Path
import traceback
from datetime import datetime
import json

video_prediction_bp = Blueprint('video_prediction', __name__, url_prefix='/video')

# Services (initialized in app.py)
model_loader = None
minio_handler = None

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------
def init_video_routes(model_service, minio_service):
    """Initialize video routes with services"""
    global model_loader, minio_handler
    model_loader = model_service
    minio_handler = minio_service


def allowed_video(filename):
    """Check if video extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


# ---------------------------------------------------------------------
# Prediction Normalizer (CRITICAL FIX)
# ---------------------------------------------------------------------
def normalize_video_prediction(result: dict):
    """
    Normalize model output into unified prediction format
    """
    # Case 1: already structured
    if isinstance(result, dict) and 'prediction' in result:
        return result['prediction']

    # Case 2: flat output
    if 'label' in result and 'confidence' in result:
        scores = result.get('scores', {})
        real_score = scores.get('real', 1.0 - float(result['confidence']))
        fake_score = scores.get('fake', float(result['confidence']))

        return {
            'label': result['label'],
            'is_fake': result['label'].lower() == 'fake',
            'confidence': float(result['confidence']),
            'real_score': float(real_score),
            'fake_score': float(fake_score)
        }

    # probabilities
    if 'probabilities' in result:
        real_score = float(result['probabilities'][0])
        fake_score = float(result['probabilities'][1])
        label = 'fake' if fake_score > real_score else 'real'

        return {
            'label': label,
            'is_fake': label == 'fake',
            'confidence': max(real_score, fake_score),
            'real_score': real_score,
            'fake_score': fake_score
        }

    raise ValueError(f"Unsupported model output format: {result}")


# ---------------------------------------------------------------------
# Upload Endpoint
# ---------------------------------------------------------------------
@video_prediction_bp.route('/upload', methods=['POST'])
def upload_video():
    """
    Upload a video to MinIO
    ---
    tags:
      - Video Upload
    consumes:
      - multipart/form-data
    parameters:
      - name: video
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Upload successful
    """
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400

        file = request.files['video']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        if not allowed_video(file.filename):
            return jsonify({
                'success': False,
                'error': f'Video type not allowed. Allowed: {", ".join(ALLOWED_VIDEO_EXTENSIONS)}'
            }), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        object_name = f"video_{timestamp}_{filename}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            video_url = minio_handler.upload_video(
                file_path=tmp_path,
                object_name=object_name,
                content_type=file.mimetype or 'video/mp4'
            )

            return jsonify({
                'success': True,
                'filename': filename,
                'video_url': video_url,
                'timestamp': datetime.now().isoformat()
            })

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ---------------------------------------------------------------------
# Predict Endpoint (FIXED)
# ---------------------------------------------------------------------
@video_prediction_bp.route('/predict', methods=['POST'])
def predict_video():
    """
    Predict if a video is real or fake with frame-level details
    ---
    tags:
      - Video Prediction
    consumes:
      - multipart/form-data
    parameters:
      - name: video
        in: formData
        type: file
        required: true
      - name: sample_rate
        in: formData
        type: integer
        default: 30
      - name: batch_size
        in: formData
        type: integer
        default: 8
      - name: aggregation
        in: formData
        type: string
        default: mean
    """
    try:
        # --------------------------------------------------
        # 1. Validate input
        # --------------------------------------------------
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400

        file = request.files['video']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        if not allowed_video(file.filename):
            return jsonify({
                'success': False,
                'error': f'Video type not allowed. Allowed: {", ".join(ALLOWED_VIDEO_EXTENSIONS)}'
            }), 400

        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > MAX_VIDEO_SIZE:
            return jsonify({'success': False, 'error': 'Video exceeds 500MB'}), 400

        # --------------------------------------------------
        # 2. Parameters
        # --------------------------------------------------
        sample_rate = int(request.form.get('sample_rate', 30))
        batch_size = int(request.form.get('batch_size', 8))
        max_frames = request.form.get('max_frames', type=int)
        aggregation = request.form.get('aggregation', 'mean')

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_filename = f"result_{timestamp}_{filename}.json"

        # --------------------------------------------------
        # 3. Save temp video
        # --------------------------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            print(f"\n[VIDEO PREDICT] Processing: {filename}")

            # Run prediction
            result = model_loader.predict_video(
                video_path=tmp_path,
                sample_rate=sample_rate,
                batch_size=batch_size,
                max_frames=max_frames,
                aggregation=aggregation
            )

            # Extract data
            prediction = result['video_prediction']
            video_metadata = result.get('video_metadata', {})
            frame_statistics = result.get('frame_statistics', {})
            frame_predictions = result.get('frame_predictions', [])

            # --------------------------------------------------
            # 4. Format frame predictions (like image route)
            # --------------------------------------------------
            frames_with_labels = []
            for i, frame_pred in enumerate(frame_predictions):
                frames_with_labels.append({
                    'frame_number': i + 1,
                    'frame_index': frame_pred.get('frame_index', i * sample_rate),
                    'timestamp': round(frame_pred.get('frame_index', i * sample_rate) / video_metadata.get('fps', 30), 2),
                    'label': 'FAKE' if frame_pred.get('is_fake', False) else 'REAL',
                    'is_fake': frame_pred.get('is_fake', False),
                    'confidence': round(float(frame_pred.get('confidence', 0)), 4),
                    'scores': {
                        'real': round(float(frame_pred.get('real_score', 0)), 4),
                        'fake': round(float(frame_pred.get('fake_score', 0)), 4)
                    }
                })

            # --------------------------------------------------
            # 5. Prepare full result for MinIO upload
            # --------------------------------------------------
            full_result = {
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                'prediction': {
                    'label': prediction['label'],
                    'is_fake': prediction['is_fake'],
                    'confidence': round(float(prediction['confidence']), 4),
                    'scores': {
                        'real': round(float(prediction['real_score']), 4),
                        'fake': round(float(prediction['fake_score']), 4)
                    }
                },
                'video_metadata': {
                    'filename': filename,
                    'duration_seconds': video_metadata.get('duration_seconds'),
                    'fps': video_metadata.get('fps'),
                    'resolution': f"{video_metadata.get('width')}x{video_metadata.get('height')}",
                    'total_frames': video_metadata.get('total_frames'),
                    'frames_analyzed': video_metadata.get('frames_extracted')
                },
                'frame_statistics': {
                    'total_analyzed': frame_statistics.get('total_analyzed'),
                    'fake_frames': frame_statistics.get('fake_frames'),
                    'real_frames': frame_statistics.get('real_frames'),
                    'fake_percentage': frame_statistics.get('fake_percentage'),
                    'real_percentage': frame_statistics.get('real_percentage'),
                    'average_confidence': round(float(frame_statistics.get('average_confidence', 0)), 4)
                },
                'frames': frames_with_labels,
                'aggregation_method': result.get('aggregation_method')
            }

            # --------------------------------------------------
            # 6. Upload full result to MinIO
            # --------------------------------------------------
            print("  Uploading result to MinIO...")
            result_url = minio_handler.upload_result_json(
                data=full_result,
                result_filename=result_filename
            )
            print(f"  Uploaded result: {result_filename}")

            print(f"  Complete! Result: {prediction['label']} ({prediction['confidence']:.2%})")

            # --------------------------------------------------
            # 7. Return response (similar to image route)
            # --------------------------------------------------
            return jsonify({
                'success': True,
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                
                # Main prediction
                'prediction': {
                    'label': prediction['label'],
                    'is_fake': prediction['is_fake'],
                    'confidence': round(float(prediction['confidence']), 4),
                    'scores': {
                        'real': round(float(prediction['real_score']), 4),
                        'fake': round(float(prediction['fake_score']), 4)
                    }
                },

                # Video metadata
                'video_metadata': {
                    'filename': filename,
                    'duration_seconds': video_metadata.get('duration_seconds'),
                    'fps': video_metadata.get('fps'),
                    'resolution': f"{video_metadata.get('width')}x{video_metadata.get('height')}",
                    'total_frames': video_metadata.get('total_frames'),
                    'frames_analyzed': video_metadata.get('frames_extracted')
                },

                # Frame statistics
                'frame_statistics': {
                    'total_analyzed': frame_statistics.get('total_analyzed'),
                    'fake_frames': frame_statistics.get('fake_frames'),
                    'real_frames': frame_statistics.get('real_frames'),
                    'fake_percentage': frame_statistics.get('fake_percentage'),
                    'real_percentage': frame_statistics.get('real_percentage'),
                    'average_confidence': round(float(frame_statistics.get('average_confidence', 0)), 4)
                },

                # Frame-level predictions (NEW!)
                'frames': frames_with_labels,

                # Storage
                'result_url': result_url,
                'result_filename': result_filename,
                'aggregation_method': result.get('aggregation_method'),
                
                'note': 'Save result_url to database for later retrieval. Each frame includes label and confidence scores.'
            })

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        print(f"\n[ERROR] Video prediction failed: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500