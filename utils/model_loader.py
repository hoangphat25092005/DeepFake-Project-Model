import sys
import os
import torch
import torchvision.transforms as transforms


sys.path.insert(0, "/mnt/mmlab2024nas/danh/phatlh/D3")

from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer

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

            #Load checkpoint
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
            
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