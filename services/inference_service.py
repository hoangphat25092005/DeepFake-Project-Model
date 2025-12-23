import torch
import numpy as np
from PIL import Image
import io
from torchvision import transforms
from typing import Dict, List, Union, Optional
import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from api.services.preprocess_ckpt import PreprocessCheckpoint

class D3ModelService:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model_version = 'v1.0'

    def set_model_version(self, version: str):
        self.model_version = version

    def get_model_version(self) -> str:
        return self.model_version

    def load_model(self, checkpoint_path: str, model_name: str = 'd3net'):
        try:
            preprocessor = PreprocessCheckpoint.from_file(checkpoint_path)
            state_dict = preprocessor.remove_module_prefix()

            self.model = get_model(model_name)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()

            if self.device.type == 'cuda':
                self.model = self.model.half()
                torch.cuda.empty_cache()

            return True
        
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
        
    def preprocess_image(self, image_data: Union[bytes, Image.Image]) -> torch.Tensor: 
        try:
            # Convert bytes to PIL Image 
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                image = image_data.convert('RGB')

            # Apply transformations
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            if self.device.type == 'cuda':
                image_tensor = image_tensor.half()
            return image_tensor
        
        except Exception as e:
            raise Exception(f"Error in image preprocessing: {e}")
        
    def predict_single(self, image_data: Union[bytes, Image.Image]) -> Dict:
        if self.model is None:
            raise Exception("Model is not loaded")
        try:
            image_tensor = self.preprocess_image(image_data=image_data)
            # Run inference
            with torch.no_grad():
                output = self.model(image_tensor)
                prob = torch.softmax(output, dim=1)
                prediction = torch.argmax(prob, dim=1).item()
                confidence = prob[0][prediction].item()

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'real': prob[0][0].item(),
                    'fake': prob[0][1].item()
                },
                'fake_score': prob[0][1].item(),
                'real_score': prob[0][0].item(),
                'is_fake': prediction == 1
            }
        
        except Exception as e:
            raise Exception(f"Error during prediction: {e}")

    def predict_batch(self, images_data: List[Union[bytes, Image.Image]], batch_size: int) -> List[Dict]:
        if self.model is None:
            raise Exception("Model is not loaded")
        
        result = []
        try:
            for i in range(0, len(images_data), batch_size):
                batch = images_data[i:i+batch_size]

                # Preprocess batch
                batch_tensors = [self.preprocess_image(image_data=img) for img in batch]
                batch_tensor = torch.cat(batch_tensors, dim=0)

                # Run inference
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probs, dim=1)

                # Process results
                for j in range(len(batch)):
                    pred = predictions[j].item()
                    conf = probs[j][pred].item()
                    result.append({
                        'prediction': pred,
                        'confidence': conf,
                        'probabilities': {
                            'real': probs[j][0].item(),
                            'fake': probs[j][1].item()
                        },
                        'fake_score': probs[j][1].item(),
                        'real_score': probs[j][0].item(),
                        'is_fake': pred == 1
                    })

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            return result
    
        except Exception as e:
            raise Exception(f"Error during batch prediction: {e}")

    def extract_frames(
        self, 
        video_path: str, 
        sample_rate: int = 30,
        max_frames: Optional[int] = None
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        frames = []
        frame_indices = []
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
                frame_indices.append(current_frame)
                if max_frames and len(frames) >= max_frames:
                    break
            current_frame += 1
        cap.release()

        metadata = {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration_seconds': duration,
            'frames_extracted': len(frames),
            'frame_indices': frame_indices
        }
        return frames, metadata

    def predict_video(
        self,
        video_path: str,
        sample_rate: int = 30,
        max_frames: Optional[int] = None,
        aggregation: str = 'mean',
        batch_size: int = 8
    ) -> Dict:
        """
        Predict if video is fake or real
        
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame
            max_frames: Maximum frames to process
            aggregation: Aggregation method ('mean', 'median', 'max', 'voting')
            batch_size: Batch size for frame prediction
            
        Returns:
            Video prediction result dictionary
        """
        print(f"\n[VIDEO PREDICTION] Processing video...")
        
        # Extract frames
        frames, metadata = self.extract_frames(video_path, sample_rate, max_frames)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Run predictions on all frames using predict_batch
        print(f"  Running inference on {len(frames)} frames...")
        predictions = self.predict_batch(frames, batch_size=batch_size)
        
        print(f"Processed {len(frames)} frames")
        
        # Aggregate predictions
        result = self._aggregate_predictions(predictions, aggregation)
        
        # Add metadata
        result['video_metadata'] = metadata
        result['aggregation_method'] = aggregation
        
        print(f"Video Analysis Complete!")
        print(f"Result: {result['video_prediction']['label']}")
        print(f"Confidence: {result['video_prediction']['confidence']:.2%}")

        return result
    
    def _aggregate_predictions(self, predictions: List[Dict], method: str) -> Dict:
        """Aggregate frame predictions into video-level prediction"""
        
        fake_scores = [p['fake_score'] for p in predictions]
        real_scores = [p['real_score'] for p in predictions]
        
        if method == 'mean':
            avg_fake = np.mean(fake_scores)
            avg_real = np.mean(real_scores)
        elif method == 'median':
            avg_fake = np.median(fake_scores)
            avg_real = np.median(real_scores)
        elif method == 'max':
            avg_fake = np.max(fake_scores)
            avg_real = 1 - avg_fake
        elif method == 'voting':
            fake_votes = sum(1 for p in predictions if p['is_fake'])
            avg_fake = fake_votes / len(predictions)
            avg_real = 1 - avg_fake
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        is_fake = avg_fake > avg_real
        confidence = max(avg_fake, avg_real)
        
        # Statistics
        fake_count = sum(1 for p in predictions if p['is_fake'])
        real_count = len(predictions) - fake_count
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        return {
            'video_prediction': {
                'label': 'FAKE' if is_fake else 'REAL',
                'is_fake': is_fake,
                'confidence': round(float(confidence), 4),
                'fake_score': round(float(avg_fake), 4),
                'real_score': round(float(avg_real), 4)
            },
            'frame_statistics': {
                'total_analyzed': len(predictions),
                'fake_frames': fake_count,
                'real_frames': real_count,
                'fake_percentage': round(fake_count / len(predictions) * 100, 2),
                'real_percentage': round(real_count / len(predictions) * 100, 2),
                'average_confidence': round(float(avg_confidence), 4)
            }
        }
        
    def upload_model(self, checkpoint_path: str, model_name: str = 'd3net'):
        if self.model is not None:
            del self.model
            self.model = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return self.load_model(checkpoint_path, model_name)