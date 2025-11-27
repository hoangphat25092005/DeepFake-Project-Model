import torch
import numpy as np
from PIL import Image
import io
from torchvision import transforms
from typing import Dict, List, Union
import sys
import os

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

    def load_model(self, checkpoint_path: str, model_name: str = 'd3net'):
        try:
            preprocessor = PreprocessCheckpoint.from_file(checkpoint_path)
            state_dict = preprocessor.remove_module_prefix()

            self.model = get_model(model_name)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()

            if self.device == 'cuda':
                self.model = self.model.half()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            return True
        
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
        
    def preprocess_image(self, image_data: Union[bytes, Image.Image]) -> torch.tensor: 
        try:
            #Convert bytes to PIL Image 
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                image = image_data.convert('RGB')

            #Apply transformations
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            if self.device == 'cuda':
                image_tensor = image_tensor.half()
            return image_tensor
        
        except Exception as e:
            raise Exception(f"Error in image preprocessing: {e}")
        
    def predict_single(self, image_data: Union[bytes, Image.Image]) -> Dict:
        if self.model is None:
            raise Exception("Model is not loaded")
        try:
            image_tensor = self.preprocess_image(image_data=image_data)
            #Run inference
            with torch.no_grad():
                output = self.model(image_tensor)
                prob = torch.softmax(output, dim=1)
                prediction = torch.argmax(prob, dim=1).item()
                confidence = prob[0][prediction].item()

            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'real': prob[0][0].item(),
                    'fake': prob[0][1].item()
                }
            }
        
        except Exception as e:
            raise Exception(f"Error during prediction: {e}")

    def predict_batch(self, images_data: List[Union[bytes, Image.Image]], batch_size: int) -> List[Dict]:
        if self.model is None:
            raise Exception("Model is not loaded")
        
        result = []

        try:
            for i in range(len(images_data)):
                batch = images_data[i:i+batch_size]

                #preprocess batch
                batch_tensors = [self.preprocess_image(image_data=img) for img in batch]
                batch_tensor = torch.cat(batch_tensors, dim=0)

                #Run inference
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probs, dim=1)

                #Process results
                for j in range(len(batch)):
                    pred = predictions[j].item()
                    conf = probs[j][pred].item()
                    result.append({
                        'prediction': pred,
                        'confidence': conf,
                        'probabilities': {
                            'real': probs[j][0].item(),
                            'fake': probs[j][1].item()
                        }
                    })

                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            return result
    
        except Exception as e:
            raise Exception(f"Error during batch prediction: {e}")
        
    
    def upload_model(self, checkpoint_path: str, model_name: str = 'd3net'):
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    