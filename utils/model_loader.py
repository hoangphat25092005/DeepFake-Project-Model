import sys
import os
import io
import cv2
import csv
import time
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# Try to load real D3 model, fallback to stub if not available
try:
    from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer
    print("[INFO] Loaded real D3 CLIP model from models.clip_models")
except (ModuleNotFoundError, ImportError) as e:
    print(f"[WARNING] Could not load real D3 model: {e}")
    print("[WARNING] Attempting to use stub model instead")
    try:
        from models_stub.clip_models import CLIPModelShuffleAttentionPenultimateLayer
        print("[INFO] Using stub D3 CLIP model from models_stub")
    except (ModuleNotFoundError, ImportError) as stub_error:
        print(f"[ERROR] Could not load stub model either: {stub_error}")
        raise


class D3ModelLoader:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

        print(f"[Info Initializing D3] device ...")
        print(f"Device: {self.device}")
        print(f"Checkpoint path: {self.checkpoint_path}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        self.model = self._load_model()
        print(f"[Info Initializing D3] model loaded successfully.")

    def _load_model(self):
        try:
            model = CLIPModelShuffleAttentionPenultimateLayer(
                "ViT-L/14",
                shuffle_times=1,
                original_times=1,
                patch_size=[14]
            )

            # Load checkpoint if path is provided and exists
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                print(f"[Info Loading D3] Loading checkpoint from {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

                #Detect checkpoint type and load state dict accordingly
                has_model_prefix = any(key.startswith('model.') for key in checkpoint.keys())
                has_attention_head = any(key.startswith('model.attention_head.') for key in checkpoint.keys())

                if has_model_prefix and has_attention_head:
                    print(f"[Info Loading D3] Loading full model from checkpoint with 'model.' prefix.")
                    model.load_state_dict(checkpoint, strict=True)

                elif has_attention_head:
                    print("Loading attention head checkpoint ")
                    model.load_state_dict(checkpoint, strict=False)

                else:
                    print(f"[Info Loading D3] Loading checkpoint without 'model.' prefix.")
                    model.load_state_dict(checkpoint, strict=False)
            else:
                print(f"[Info Loading D3] No checkpoint path provided, using model with default weights")

            model = model.to(self.device)
            model.eval()

            return model
        
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
        
    def preprocess_image(self, image):
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        return image_tensor
        
    def predict(self, image):
        try:
            image_tensor = self.preprocess_image(image)

            #run inference
            with torch.no_grad():
                output = self.model(image_tensor)

                if len(output.shape) > 1 and output.shape[-1] == 2:
                    prob = torch.softmax(output, dim=1)
                    fake_prob = prob[0][1].item()

                elif len(output.shape) > 1 and output.shape[-1] == 1:
                    fake_prob = torch.sigmoid(output.squeeze(-1))[0].item()
                
                else:
                    fake_prob = torch.sigmoid(output)[0].item()
                
                real = 1 - fake_prob

                is_fake = fake_prob > 0.5
                confidence = fake_prob if is_fake else real


                return {
                    "is_fake": bool(is_fake),
                    "label": 'FAKE' if is_fake else 'REAL',
                    "confidence": float(confidence),
                    "real_score": float(real),
                    "fake_score": float(fake_prob),
                    "raw_output": float(output[0].item())
                }
        except Exception as e:
            raise Exception(f"Error in image preprocessing: {e}")
        
    def predict_batch(self, images):
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
    

    def extract_frames(self, video_path, sample_rate=30, max_frames=None):
        """
        Extract frames from a video file.

        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame
            max_frames: Maximum number of frames to extract

        Returns:
            frames: List of PIL.Image frames
            metadata: Dict with video info
        """
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
        video_path,
        sample_rate=30,
        max_frames=None,
        aggregation='mean',
        batch_size=8
    ):
        """
        Predict if video is fake or real by analyzing frames.

        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame
            max_frames: Maximum frames to process
            aggregation: Aggregation method ('mean', 'median', 'max', 'voting')
            batch_size: Batch size for frame prediction

        Returns:
            Video prediction result dictionary
        """
        frames, metadata = self.extract_frames(video_path, sample_rate, max_frames)
        if not frames:
            raise ValueError("No frames extracted from video")

        # Batch prediction
        predictions = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            batch_results = self.predict_batch(batch)
            predictions.extend(batch_results)

        # Aggregate
        fake_scores = [p['fake_score'] for p in predictions]
        real_scores = [p['real_score'] for p in predictions]

        if aggregation == 'mean':
            avg_fake = np.mean(fake_scores)
            avg_real = np.mean(real_scores)
        elif aggregation == 'median':
            avg_fake = np.median(fake_scores)
            avg_real = np.median(real_scores)
        elif aggregation == 'max':
            avg_fake = np.max(fake_scores)
            avg_real = 1 - avg_fake
        elif aggregation == 'voting':
            fake_votes = sum(1 for p in predictions if p['is_fake'])
            avg_fake = fake_votes / len(predictions)
            avg_real = 1 - avg_fake
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        is_fake = avg_fake > avg_real
        confidence = max(avg_fake, avg_real)
        fake_count = sum(1 for p in predictions if p['is_fake'])
        real_count = len(predictions) - fake_count
        avg_confidence = np.mean([p['confidence'] for p in predictions])

        return {
            'video_prediction': {
                'label': 'FAKE' if is_fake else 'REAL',
                'is_fake': bool(is_fake),
                'confidence': float(confidence),
                'fake_score': float(avg_fake),
                'real_score': float(avg_real)
            },
            'video_metadata': metadata,
            'frame_statistics': {
                'total_analyzed': len(predictions),
                'fake_frames': fake_count,
                'real_frames': real_count,
                'fake_percentage': round(fake_count / len(predictions) * 100, 2),
                'real_percentage': round(real_count / len(predictions) * 100, 2),
                'average_confidence': float(avg_confidence)
            },
            'aggregation_method': aggregation
        }

    def upload_frame_results_csv(self, frame_predictions, result_filename):
        try:
            output = io.StringIO()
            writer = csv.writer(output, fieldnames=['frame_index', 'timestamp', 'prediction', 'real_score'])
            writer.writeheader()
            for row in frame_predictions:
                writer.writerow(row)
            output.seek(0)
            data = io.BytesIO(output.read(), encoding=('utf-8'))
            file_size = data.getbuffer().nbytes

            self.client.put_object(
                self.bucket_name,
                result_filename,
                data,
                file_size,
                content_type='text/csv'
            )

            url = self.client.presigned_get_object(
                self.bucket_name,
                result_filename,
                expires=timedelta(days=7)
            )
            print(f"Uploaded frame results CSV: {result_filename}")
            return url
        except S3Error as e:
            print(f"Upload error: {e}")
            raise Exception("Failed to upload frame results CSV to MinIO")