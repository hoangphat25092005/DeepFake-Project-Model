#!/usr/bin/env python3
"""
Visualize Ground Truth vs Prediction for D3 Deepfake Detection
Shows side-by-side comparison with correctness indicators
"""

import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os
import glob
from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GTvsPredVisualizer:
    """Visualize Ground Truth vs Prediction"""
    
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
        self.output_dir = "/mnt/mmlab2024nas/danh/phatlh/D3/inference_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the D3 model with proper checkpoint handling"""
        try:
            print(f"Loading model from {self.checkpoint_path}")
            
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
                # Full model checkpoint (from training) - contains both CLIP and attention head
                print("✓ Detected: Full model checkpoint (CLIP + attention head)")
                self.model.load_state_dict(checkpoint, strict=False)
                print("✓ Loaded full model successfully")
                
            elif has_attention_prefix:
                # Checkpoint with only attention_head prefix
                print("✓ Detected: Attention head checkpoint with prefix")
                self.model.load_state_dict(checkpoint, strict=False)
                print("✓ Loaded attention head successfully")
                
            else:
                # Only attention head weights without prefix
                print("✓ Detected: Attention head checkpoint without prefix")
                attention_state_dict = {f'attention_head.{k}': v for k, v in checkpoint.items()}
                missing, unexpected = self.model.load_state_dict(attention_state_dict, strict=False)
                print(f"✓ Loaded attention head successfully")
                if missing:
                    print(f"  Note: {len(missing)} keys loaded from pretrained CLIP")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded and ready for inference!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
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
                    # If output has 2 classes, use softmax
                    prob = torch.softmax(output, dim=1)
                    fake_prob = prob[0][1].item()
                elif len(output.shape) > 1 and output.shape[-1] == 1:
                    # Single output with extra dimension, use sigmoid
                    fake_prob = torch.sigmoid(output.squeeze(-1))[0].item()
                else:
                    # Single output, apply sigmoid
                    fake_prob = torch.sigmoid(output)[0].item()
                
                pred_label = 1 if fake_prob > 0.5 else 0
            
            return {
                'image': original_image,
                'pred_label': pred_label,
                'pred_name': 'FAKE' if pred_label == 1 else 'REAL',
                'confidence': fake_prob if pred_label == 1 else (1 - fake_prob),
                'fake_score': fake_prob
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def visualize_gt_vs_pred(self, data_dir, max_images=16, images_per_row=4):
        """
        Create GT vs Prediction visualization
        
        Args:
            data_dir: Path to folder with 0_real and 1_fake subfolders
            max_images: Maximum number of images to visualize
            images_per_row: Number of images per row
        """
        print(f"\n{'='*70}")
        print(f"GROUND TRUTH vs PREDICTION VISUALIZATION")
        print(f"{'='*70}")
        print(f"Data directory: {data_dir}")
        
        # Get images from both classes
        real_dir = os.path.join(data_dir, "0_real")
        fake_dir = os.path.join(data_dir, "1_fake")
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            print(f"ERROR: Expected structure:")
            print(f"{data_dir}/0_real/")
            print(f"{data_dir}/1_fake/")
            return None
        
        real_images = glob.glob(os.path.join(real_dir, "*.jpg")) + \
                      glob.glob(os.path.join(real_dir, "*.png")) + \
                      glob.glob(os.path.join(real_dir, "*.jpeg"))
        
        fake_images = glob.glob(os.path.join(fake_dir, "*.jpg")) + \
                      glob.glob(os.path.join(fake_dir, "*.png")) + \
                      glob.glob(os.path.join(fake_dir, "*.jpeg"))
        
        print(f"Found {len(real_images)} real images")
        print(f"Found {len(fake_images)} fake images")
        
        # Balance the dataset - take equal number from each class
        n_per_class = min(len(real_images), len(fake_images), max_images // 2)
        
        real_images = real_images[:n_per_class]
        fake_images = fake_images[:n_per_class]
        
        # Combine and create ground truth labels
        all_images = []
        ground_truths = []
        
        for img in real_images:
            all_images.append(img)
            ground_truths.append((0, 'REAL'))
        
        for img in fake_images:
            all_images.append(img)
            ground_truths.append((1, 'FAKE'))
        
        total = len(all_images)
        print(f"\nProcessing {total} images ({n_per_class} real + {n_per_class} fake)...")
        
        # Process all images
        results = []
        correct = 0
        
        for i, (img_path, (gt_label, gt_name)) in enumerate(zip(all_images, ground_truths), 1):
            print(f"   [{i}/{total}] Processing: {os.path.basename(img_path)[:30]}...", end='')
            
            pred = self.predict_image(img_path)
            if pred is None:
                print(" Failed")
                continue
            
            is_correct = (pred['pred_label'] == gt_label)
            if is_correct:
                correct += 1
            
            results.append({
                'image': pred['image'],
                'gt_label': gt_label,
                'gt_name': gt_name,
                'pred_label': pred['pred_label'],
                'pred_name': pred['pred_name'],
                'confidence': pred['confidence'],
                'fake_score': pred['fake_score'],
                'is_correct': is_correct,
                'filename': os.path.basename(img_path)
            })
            
            status = "✓" if is_correct else "✗"
            print(f" {status} GT:{gt_name} Pred:{pred['pred_name']} ({pred['confidence']:.3f})")
        
        accuracy = (correct / len(results)) * 100 if results else 0
        print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.2f}%")
        
        # Create visualization
        self.create_grid_visualization(results, images_per_row)
        
        # Create detailed statistics
        self.create_statistics_report(results)
        
        return results
    
    def create_grid_visualization(self, results, images_per_row=4):
        """Create a grid showing GT vs Prediction"""
        print(f"\nCreating grid visualization...")
        
        n_images = len(results)
        n_rows = (n_images + images_per_row - 1) // images_per_row
        
        # Create figure with larger size for better readability
        fig = plt.figure(figsize=(images_per_row * 4, n_rows * 4.5))
        
        for idx, result in enumerate(results):
            ax = plt.subplot(n_rows, images_per_row, idx + 1)
            
            # Display image
            ax.imshow(result['image'])
            
            # Determine color based on correctness
            is_correct = result['is_correct']
            border_color = 'green' if is_correct else 'red'
            symbol = '✓' if is_correct else '✗'
            
            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4)
            
            # Create title with GT and Prediction
            title = f"{symbol} {result['filename'][:20]}\n"
            title += f"GT: {result['gt_name']}  |  "
            title += f"Pred: {result['pred_name']}\n"
            title += f"Confidence: {result['confidence']:.3f}"
            
            ax.set_title(title, fontsize=10, weight='bold', color=border_color, pad=10)
            ax.axis('off')
        
        plt.suptitle('Ground Truth vs Prediction', fontsize=16, weight='bold', y=0.995)
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(self.output_dir, f"gt_vs_pred_grid_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grid visualization saved: {save_path}")
    
    def create_statistics_report(self, results):
        """Create detailed statistics report"""
        print(f"\nCreating statistics report...")
        
        # Calculate metrics
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        incorrect = total - correct
        
        # Confusion matrix
        tp = sum(1 for r in results if r['gt_label'] == 1 and r['pred_label'] == 1)
        tn = sum(1 for r in results if r['gt_label'] == 0 and r['pred_label'] == 0)
        fp = sum(1 for r in results if r['gt_label'] == 0 and r['pred_label'] == 1)
        fn = sum(1 for r in results if r['gt_label'] == 1 and r['pred_label'] == 0)
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        confusion = np.array([[tn, fp], [fn, tp]])
        im = ax1.imshow(confusion, cmap='Blues', aspect='auto')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, confusion[i, j],
                              ha="center", va="center", color="black", fontsize=14, weight='bold')
        
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Real', 'Fake'])
        ax1.set_yticklabels(['Real', 'Fake'])
        ax1.set_xlabel('Predicted', fontsize=11, weight='bold')
        ax1.set_ylabel('Ground Truth', fontsize=11, weight='bold')
        ax1.set_title('Confusion Matrix', fontsize=12, weight='bold')
        plt.colorbar(im, ax=ax1)
        
        # 2. Accuracy Pie Chart
        ax2 = fig.add_subplot(gs[0, 1])
        if correct > 0 or incorrect > 0:
            colors = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0.05)
            ax2.pie([correct, incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%',
                   colors=colors, explode=explode, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
            ax2.set_title(f'Overall Accuracy: {accuracy:.2f}%', fontsize=12, weight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, weight='bold')
            ax2.set_title('Overall Accuracy: N/A', fontsize=12, weight='bold')
            ax2.axis('off')
        
        # 3. Metrics Bar Chart
        ax3 = fig.add_subplot(gs[0, 2])
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [accuracy, precision * 100, recall * 100, f1 * 100]
        bars = ax3.bar(metrics_names, metrics_values, color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c'])
        ax3.set_ylim(0, 100)
        ax3.set_ylabel('Score (%)', fontsize=11, weight='bold')
        ax3.set_title('Performance Metrics', fontsize=12, weight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 4. Confidence Distribution for Correct Predictions
        ax4 = fig.add_subplot(gs[1, :2])
        correct_confidences = [r['confidence'] for r in results if r['is_correct']]
        if correct_confidences:
            ax4.hist(correct_confidences, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(correct_confidences), color='darkgreen', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(correct_confidences):.3f}')
            ax4.set_xlabel('Confidence Score', fontsize=11, weight='bold')
            ax4.set_ylabel('Frequency', fontsize=11, weight='bold')
            ax4.set_title('Confidence Distribution (Correct Predictions)', fontsize=12, weight='bold')
            ax4.legend()
            ax4.grid(alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Correct Predictions', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12, weight='bold')
            ax4.set_title('Confidence Distribution (Correct Predictions)', fontsize=12, weight='bold')
        
        # 5. Confidence Distribution for Incorrect Predictions
        ax5 = fig.add_subplot(gs[1, 2])
        incorrect_confidences = [r['confidence'] for r in results if not r['is_correct']]
        if incorrect_confidences:
            ax5.hist(incorrect_confidences, bins=10, color='red', alpha=0.7, edgecolor='black')
            ax5.axvline(np.mean(incorrect_confidences), color='darkred', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(incorrect_confidences):.3f}')
            ax5.set_xlabel('Confidence Score', fontsize=11, weight='bold')
            ax5.set_ylabel('Frequency', fontsize=11, weight='bold')
            ax5.set_title('Confidence Distribution (Incorrect)', fontsize=12, weight='bold')
            ax5.legend()
            ax5.grid(alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No Incorrect Predictions', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12, weight='bold')
            ax5.set_title('Confidence Distribution (Incorrect)', fontsize=12, weight='bold')
        
        # 6. Statistics Text
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        stats_text = f"""
        DETAILED STATISTICS REPORT
        
        Total Images Evaluated: {total}
        ✓ Correct Predictions: {correct} ({accuracy:.2f}%)
        ✗ Incorrect Predictions: {incorrect} ({100-accuracy:.2f}%)
        
        Confusion Matrix Breakdown:
        • True Positives (Fake → Fake): {tp}
        • True Negatives (Real → Real): {tn}
        • False Positives (Real → Fake): {fp}
        • False Negatives (Fake → Real): {fn}
        
        Performance Metrics:
        • Precision: {precision:.4f} ({precision*100:.2f}%)
        • Recall: {recall:.4f} ({recall*100:.2f}%)
        • F1-Score: {f1:.4f} ({f1*100:.2f}%)
        
        Confidence Statistics:
        • Correct Predictions: Mean = {np.mean(correct_confidences) if correct_confidences else 0:.3f}, Std = {np.std(correct_confidences) if correct_confidences else 0:.3f}
        • Incorrect Predictions: Mean = {np.mean(incorrect_confidences) if incorrect_confidences else 0:.3f}, Std = {np.std(incorrect_confidences) if incorrect_confidences else 0:.3f}
        """
        
        ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('D³ Model Performance Analysis', fontsize=16, weight='bold', y=0.995)
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(self.output_dir, f"statistics_report_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Statistics report saved: {save_path}")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("D³ DEEPFAKE DETECTION - GT vs PREDICTION VISUALIZATION")
    print("="*70)
    
    # Configuration - UPDATE THESE PATHS
    checkpoint_path = "/mnt/mmlab2024nas/danh/phatlh/D3/checkpoints/finetune_wildrf/model_epoch_best.pth"  # Your fine-tuned model
    data_dir = "/mnt/mmlab2024nas/danh/phatlh/D3/data/WildRF/test/twitter"  # Directory with 0_real and 1_fake subfolders
    
    # Check paths
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print(f"Using pretrained checkpoint instead...")
        checkpoint_path = "/mnt/mmlab2024nas/danh/phatlh/D3/ckpt/classifier.pth"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    # Initialize visualizer
    try:
        visualizer = GTvsPredVisualizer(checkpoint_path)
    except Exception as e:
        print(f"Failed to initialize visualizer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create visualizations
    try:
        results = visualizer.visualize_gt_vs_pred(
            data_dir=data_dir,
            max_images=32,  # Change this to visualize more/fewer images
            images_per_row=4
        )
        
        if results:
            print(f"\n{'='*70}")
            print(f"VISUALIZATION COMPLETED SUCCESSFULLY!")
            print(f"Output directory: {visualizer.output_dir}")
            print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()