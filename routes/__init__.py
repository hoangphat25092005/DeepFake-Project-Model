from .health_check_route import health_bp
from .prediction_route import prediction_bp
from .video_prediction_route import video_prediction_bp

__all__ = ['health_bp', 'prediction_bp', 'video_prediction_bp']
