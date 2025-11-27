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