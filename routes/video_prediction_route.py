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

    # Case 3: probabilities
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
    Predict if a video is real or fake
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
        frame_json_name = f"frames_{timestamp}_{filename}.json"

        # --------------------------------------------------
        # 3. Save temp video
        # --------------------------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            print(f"\n[VIDEO] Processing: {filename}")

            result = model_loader.predict_video(
                video_path=tmp_path,
                sample_rate=sample_rate,
                batch_size=batch_size,
                max_frames=max_frames,
                aggregation=aggregation
            )

            # DEBUG (remove later)
            print("MODEL OUTPUT:", json.dumps(result, indent=2, default=str))

            prediction = normalize_video_prediction(result)

            # --------------------------------------------------
            # 4. Frame results upload
            # --------------------------------------------------
            frame_predictions = result.get('frame_predictions', [])
            frame_results_url = None

            if frame_predictions:
                frame_results_url = minio_handler.upload_frame_results_json(
                    frame_predictions=frame_predictions,
                    result_filename=frame_json_name
                )

            # --------------------------------------------------
            # 5. Response
            # --------------------------------------------------
            return jsonify({
                'success': True,
                'filename': filename,
                'timestamp': datetime.now().isoformat(),

                'prediction': {
                    'label': prediction['label'],
                    'is_fake': prediction['is_fake'],
                    'confidence': round(prediction['confidence'], 4),
                    'scores': {
                        'real': round(prediction['real_score'], 4),
                        'fake': round(prediction['fake_score'], 4)
                    }
                },

                'video_info': {
                    'duration_sec': result.get('video_metadata', {}).get('duration'),
                    'fps': result.get('video_metadata', {}).get('fps'),
                    'total_frames': result.get('video_metadata', {}).get('total_frames'),
                    'processed_frames': result.get('frame_statistics', {}).get('processed')
                },

                'frame_results_url': frame_results_url,
                'aggregation_method': result.get('aggregation_method', aggregation),
                'note': 'Frame-level predictions are stored as JSON for later retrieval'
            })

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
