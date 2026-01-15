#!/usr/bin/env python3
"""
Test D3 Deepfake Detector on a single image or folder of images
"""

import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob
from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')


class SingleImageDetector:
    """Test single images with D3 detector"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
        
        # Output directory
        self.output_dir = "/mnt/mmlab2024nas/danh/phatlh/D3/single_image_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the D3 model"""
        try:
            print(f"\nLoading model from: {self.checkpoint_path}")
            
            # Initialize model
            self.model = CLIPModelShuffleAttentionPenultimateLayer(
                "ViT-L/14", 
                shuffle_times=1, 
                original_times=1, 
                patch_size=[14]
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Detect checkpoint type
            has_model_prefix = any(k.startswith('model.') for k in checkpoint.keys())
            has_attention_prefix = any(k.startswith('attention_head.') for k in checkpoint.keys())
            
            if has_model_prefix and has_attention_prefix:
                print("✓ Full model checkpoint detected")
                self.model.load_state_dict(checkpoint, strict=False)
            elif has_attention_prefix:
                print("✓ Attention head checkpoint (with prefix) detected")
                self.model.load_state_dict(checkpoint, strict=False)
            else:
                print("✓ Attention head checkpoint (without prefix) detected")
                attention_state_dict = {f'attention_head.{k}': v for k, v in checkpoint.items()}
                self.model.load_state_dict(attention_state_dict, strict=False)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully!\n")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_image(self, image_path):
        """Predict if an image is fake or real"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
            
            # Transform
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # Handle different output formats
                if len(output.shape) > 1 and output.shape[-1] == 2:
                    prob = torch.softmax(output, dim=1)
                    fake_prob = prob[0][1].item()
                elif len(output.shape) > 1 and output.shape[-1] == 1:
                    fake_prob = torch.sigmoid(output.squeeze(-1))[0].item()
                else:
                    fake_prob = torch.sigmoid(output)[0].item()
                
                pred_label = 1 if fake_prob > 0.5 else 0
            
            return {
                'image': original_image,
                'pred_label': pred_label,
                'pred_name': 'FAKE' if pred_label == 1 else 'REAL',
                'confidence': fake_prob if pred_label == 1 else (1 - fake_prob),
                'fake_score': fake_prob,
                'real_score': 1 - fake_prob
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def visualize_single_prediction(self, image_path):
        """Create visualization for a single image"""
        print(f"\n{'='*70}")
        print(f"Testing image: {os.path.basename(image_path)}")
        print(f"{'='*70}")
        
        # Get prediction
        result = self.predict_image(image_path)
        
        if result is None:
            print("Failed to process image")
            return None
        
        # Print results
        print(f"\nPREDICTION RESULTS:")
        print(f"   Prediction: {result['pred_name']}")
        print(f"   Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"   ")
        print(f"   Detailed Scores:")
        print(f"   • Real Score:  {result['real_score']:.4f} ({result['real_score']*100:.2f}%)")
        print(f"   • Fake Score:  {result['fake_score']:.4f} ({result['fake_score']*100:.2f}%)")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Original image with prediction
        ax1 = axes[0]
        ax1.imshow(result['image'])
        ax1.axis('off')
        
        # Color based on prediction
        pred_color = 'red' if result['pred_label'] == 1 else 'green'
        for spine in ax1.spines.values():
            spine.set_edgecolor(pred_color)
            spine.set_linewidth(5)
        
        title = f"Prediction: {result['pred_name']}\n"
        title += f"Confidence: {result['confidence']:.4f}"
        ax1.set_title(title, fontsize=14, weight='bold', color=pred_color, pad=20)
        
        # Right: Score bar chart
        ax2 = axes[1]
        categories = ['REAL', 'FAKE']
        scores = [result['real_score'] * 100, result['fake_score'] * 100]
        colors = ['green' if result['pred_label'] == 0 else 'lightgreen',
                  'red' if result['pred_label'] == 1 else 'lightcoral']
        
        bars = ax2.barh(categories, scores, color=colors, edgecolor='black', linewidth=2)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Score (%)', fontsize=12, weight='bold')
        ax2.set_title('Detection Scores', fontsize=14, weight='bold', pad=20)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
                    f'{score:.2f}%', ha='left', va='center',
                    fontsize=11, weight='bold')
        
        # Add threshold line
        ax2.axvline(50, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Threshold (50%)')
        ax2.legend(loc='lower right')
        
        plt.suptitle(f'D³ Deepfake Detection Result\n{os.path.basename(image_path)}',
                    fontsize=16, weight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(self.output_dir, f"{filename}_result_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved: {save_path}")
        print(f"{'='*70}\n")
        
        return result
    
    def test_folder(self, folder_path, max_images=None):
        """Test all images in a folder"""
        print(f"\n{'='*70}")
        print(f"Testing images from folder: {folder_path}")
        print(f"{'='*70}")
        
        # Get all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"\nFound {len(image_paths)} images")
        
        if len(image_paths) == 0:
            print("No images found in the folder!")
            return
        
        # Process each image
        results = []
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
            result = self.predict_image(img_path)
            
            if result:
                result['filename'] = os.path.basename(img_path)
                results.append(result)
                print(f"   → Prediction: {result['pred_name']} (confidence: {result['confidence']:.4f})")
        
        # Summary statistics
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        
        total = len(results)
        fake_count = sum(1 for r in results if r['pred_label'] == 1)
        real_count = total - fake_count
        
        print(f"Total images processed: {total}")
        print(f"Detected as REAL: {real_count} ({real_count/total*100:.1f}%)")
        print(f"Detected as FAKE: {fake_count} ({fake_count/total*100:.1f}%)")
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"Average confidence: {avg_confidence:.4f}")
        
        # Create summary visualization
        self.create_folder_summary(results, folder_path)
        
        return results
    
    def create_folder_summary(self, results, folder_path):
        """Create a summary grid for multiple images"""
        n_images = len(results)
        if n_images == 0:
            return
        
        # Calculate grid size
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(n_cols * 4, n_rows * 4.5))
        
        for idx, result in enumerate(results):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Display image
            ax.imshow(result['image'])
            
            # Color border based on prediction
            pred_color = 'red' if result['pred_label'] == 1 else 'green'
            for spine in ax.spines.values():
                spine.set_edgecolor(pred_color)
                spine.set_linewidth(4)
            
            # Title
            title = f"{result['filename'][:20]}\n"
            title += f"{result['pred_name']}\n"
            title += f"Conf: {result['confidence']:.3f}"
            
            ax.set_title(title, fontsize=10, weight='bold', color=pred_color, pad=10)
            ax.axis('off')
        
        folder_name = os.path.basename(folder_path)
        plt.suptitle(f'D³ Detection Results - {folder_name}', fontsize=16, weight='bold', y=0.995)
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(self.output_dir, f"folder_summary_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSummary visualization saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test D3 deepfake detector on single images')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--folder', type=str, help='Path to a folder of images')
    parser.add_argument('--checkpoint', type=str, 
                       default='/mnt/mmlab2024nas/danh/phatlh/D3/checkpoints/finetune_wildrf/model_epoch_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process from folder')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Using pretrained checkpoint instead...")
        args.checkpoint = "/mnt/mmlab2024nas/danh/phatlh/D3/ckpt/classifier.pth"
    
    # Initialize detector
    try:
        detector = SingleImageDetector(args.checkpoint)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return
    
    # Test single image or folder
    if args.image:
        if os.path.exists(args.image):
            detector.visualize_single_prediction(args.image)
        else:
            print(f"Image not found: {args.image}")
    
    elif args.folder:
        if os.path.exists(args.folder):
            detector.test_folder(args.folder, max_images=args.max_images)
        else:
            print(f"Folder not found: {args.folder}")
    
    else:
        print("\nPlease specify either --image or --folder")
        parser.print_help()


if __name__ == "__main__":
    main()