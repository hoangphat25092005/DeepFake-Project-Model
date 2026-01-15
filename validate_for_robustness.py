import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import Dataset
from PIL import Image 
from tqdm import tqdm
import random
import pandas as pd
from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer


def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def find_test_folders(root_dir):
    """Automatically find test data folders"""
    print(f"🔍 Searching for test data in: {root_dir}")
    
    possible_folders = []
    
    # Walk through directory to find real/fake pairs
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories and common non-data dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'api', 'models']]
        
        # Check if this folder contains real/fake subfolders
        if 'real' in dirs and 'fake' in dirs:
            real_path = os.path.join(root, 'real')
            fake_path = os.path.join(root, 'fake')
            
            # Check if folders have images
            real_images = get_image_count(real_path)
            fake_images = get_image_count(fake_path)
            
            if real_images > 0 and fake_images > 0:
                possible_folders.append({
                    'name': root.replace(root_dir, '').strip('/'),
                    'real_path': real_path,
                    'fake_path': fake_path,
                    'real_count': real_images,
                    'fake_count': fake_images
                })
        
        # Check for numbered folders (0_real, 1_fake pattern)
        real_dirs = [d for d in dirs if 'real' in d.lower() or d.endswith('_0')]
        fake_dirs = [d for d in dirs if 'fake' in d.lower() or d.endswith('_1')]
        
        for real_dir in real_dirs:
            for fake_dir in fake_dirs:
                real_path = os.path.join(root, real_dir)
                fake_path = os.path.join(root, fake_dir)
                
                real_images = get_image_count(real_path)
                fake_images = get_image_count(fake_path)
                
                if real_images > 0 and fake_images > 0:
                    possible_folders.append({
                        'name': f"{root.replace(root_dir, '').strip('/')}/{real_dir}+{fake_dir}",
                        'real_path': real_path,
                        'fake_path': fake_path,
                        'real_count': real_images,
                        'fake_count': fake_images
                    })
    
    return possible_folders


def get_image_count(folder_path):
    """Count images in a folder"""
    if not os.path.exists(folder_path):
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    count = 0
    
    try:
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                count += 1
    except PermissionError:
        return 0
    
    return count


def get_image_files(folder_path, max_files=None):
    """Get list of image files from folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = []
    
    if not os.path.exists(folder_path):
        return image_files
    
    try:
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        # Shuffle and limit
        random.shuffle(image_files)
        if max_files and len(image_files) > max_files:
            image_files = image_files[:max_files]
            
    except PermissionError:
        print(f"⚠️  Permission denied: {folder_path}")
    
    return image_files


class SimpleRealFakeDataset(Dataset):
    """Simple dataset for real/fake classification"""
    
    def __init__(self, real_path, fake_path, max_sample=None):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

        # Get image files
        real_files = get_image_files(real_path, max_sample)
        fake_files = get_image_files(fake_path, max_sample)
        
        print(f"Found {len(real_files)} real and {len(fake_files)} fake images")
        
        # Balance the dataset
        min_count = min(len(real_files), len(fake_files))
        if min_count == 0:
            print(f"   ⚠️  No valid image pairs found!")
            self.image_files = []
            self.labels = []
            return
        
        # Take equal numbers from each class
        real_files = real_files[:min_count]
        fake_files = fake_files[:min_count]
        
        # Combine and create labels
        self.image_files = real_files + fake_files
        self.labels = [0] * len(real_files) + [1] * len(fake_files)  # 0=real, 1=fake
        
        print(f"   ✅ Dataset created with {len(real_files)} real + {len(fake_files)} fake = {len(self.image_files)} total images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label = self.labels[idx]
        
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, label
        
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a dummy black image
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label


def validate_model(model, dataloader):
    """Run model validation with proper data type handling"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    print(f"🔄 Running inference on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Processing")):
            try:
                # Move to GPU but keep as Float32 (don't convert to half)
                images = images.cuda()  # Remove .half() here
                
                # Get model predictions
                outputs = model(images)
                
                # Debug: Check output type and shape
                if batch_idx == 0:
                    print(f"   Model output shape: {outputs.shape}")
                    print(f"   Model output type: {outputs.dtype}")
                
                # Convert to probabilities
                if len(outputs.shape) > 1 and outputs.shape[-1] == 2:
                    # If output has 2 classes, take the fake class probability
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                elif len(outputs.shape) > 1 and outputs.shape[-1] == 1:
                    # Single output with extra dimension
                    probs = torch.sigmoid(outputs.squeeze(-1))
                else:
                    # Single output, apply sigmoid
                    probs = torch.sigmoid(outputs)
                
                # Ensure probs is 1D
                if len(probs.shape) > 1:
                    probs = probs.flatten()
                
                # Collect predictions and labels
                all_predictions.extend(probs.cpu().float().numpy())  # Convert to float32
                all_labels.extend(labels.numpy())
                
                # Clear GPU memory
                del images, outputs, probs
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f" Error in batch {batch_idx}: {e}")
                # Print more debugging info
                if batch_idx < 3:  # Only for first few batches to avoid spam
                    print(f"   Image shape: {images.shape if 'images' in locals() else 'N/A'}")
                    print(f"   Image dtype: {images.dtype if 'images' in locals() else 'N/A'}")
                    if 'outputs' in locals():
                        print(f"   Output shape: {outputs.shape}")
                        print(f"   Output dtype: {outputs.dtype}")
                continue
    
    if len(all_predictions) == 0:
        print("No predictions generated!")
        return None
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    print(f"   Generated {len(predictions)} predictions")
    print(f"   Real samples: {np.sum(labels == 0)}")
    print(f"   Fake samples: {np.sum(labels == 1)}")
    print(f"   Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # Calculate metrics
    try:
        # Average Precision
        ap = average_precision_score(labels, predictions)
        
        # Accuracy with different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        accuracies = {}
        
        for thresh in thresholds:
            binary_preds = (predictions >= thresh).astype(int)
            acc = accuracy_score(labels, binary_preds)
            accuracies[thresh] = acc
        
        # Find best threshold
        best_thresh = max(accuracies, key=accuracies.get)
        best_acc = accuracies[best_thresh]
        
        results = {
            'ap': ap,
            'accuracies': accuracies,
            'best_threshold': best_thresh,
            'best_accuracy': best_acc,
            'predictions': predictions,
            'labels': labels
        }
        
        return results
        
    except Exception as e:
        print(f" Error calculating metrics: {e}")
        return None


def main():
    print("\n" + "="*60)
    print(" D3 Model Validation with Auto-Discovery")
    print("="*60)
    
    # Configuration
    set_seed(418)
    
    root_dir = "/mnt/mmlab2024nas/danh/phatlh/D3"
    checkpoint_path = "/mnt/mmlab2024nas/danh/phatlh/D3/ckpt/classifier.pth"
    result_folder = "/mnt/mmlab2024nas/danh/phatlh/D3/result_inference/"
    max_sample = 50  # Increase sample size
    batch_size = 2   # Reduce batch size to avoid memory issues
    
    # Create result folder
    os.makedirs(result_folder, exist_ok=True)
    
    print(f"Root directory: {root_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Max samples per class: {max_sample}")
    print(f"Batch size: {batch_size}")
    
    # Check checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Find test data
    print("\nSearching for test data...")
    test_folders = find_test_folders(root_dir)
    
    if not test_folders:
        print("No test data found!")
        print("\nPlease create test data in one of these formats:")
        print("1. folder/real/ and folder/fake/")
        print("2. folder/0_real/ and folder/1_fake/")
        print("3. Or specify manual paths in the script")
        return
    
    print(f"\nFound {len(test_folders)} test folder(s):")
    for i, folder in enumerate(test_folders):
        print(f"   {i+1}. {folder['name']}")
        print(f"      Real: {folder['real_count']} images in {folder['real_path']}")
        print(f"      Fake: {folder['fake_count']} images in {folder['fake_path']}")
    
    # Load model
    print(f"\nLoading model...")
    try:
        model = CLIPModelShuffleAttentionPenultimateLayer(
            "ViT-L/14", 
            shuffle_times=1, 
            original_times=1, 
            patch_size=[14]
        )
        
        # Load checkpoint
        print("Loading checkpoint...")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Debug: Check what's in the checkpoint
        print(f"Checkpoint keys: {list(state_dict.keys())}")
        
        model.attention_head.load_state_dict(state_dict)
        
        # Move to GPU but keep as Float32 (remove .half())
        print("Moving model to GPU...")
        model = model.cuda()  # Remove .half() here
        model.eval()
        
        print("Model loaded successfully")
        
        # Test model with dummy input
        print("🧪 Testing model with dummy input...")
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        with torch.no_grad():
            dummy_output = model(dummy_input)
            print(f"   Dummy output shape: {dummy_output.shape}")
            print(f"   Dummy output type: {dummy_output.dtype}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run validation on each test folder
    all_results = []
    
    for i, folder_info in enumerate(test_folders):
        print(f"\n📁 Testing on folder {i+1}/{len(test_folders)}: {folder_info['name']}")
        
        try:
            # Create dataset
            dataset = SimpleRealFakeDataset(
                real_path=folder_info['real_path'],
                fake_path=folder_info['fake_path'],
                max_sample=max_sample
            )
            
            if len(dataset) == 0:
                print("   ⚠️  Empty dataset, skipping...")
                continue
            
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=0
            )
            
            # Run validation
            results = validate_model(model, dataloader)
            
            if results is None:
                print("Validation failed")
                continue
            
            # Print results
            print(f"Results:")
            print(f"      Average Precision: {results['ap']:.4f}")
            print(f"      Best Accuracy: {results['best_accuracy']:.4f} @ threshold {results['best_threshold']}")
            print(f"      Accuracies: {', '.join(f'{t}: {a:.3f}' for t, a in results['accuracies'].items())}")
            
            # Store results
            folder_result = {
                'folder': folder_info['name'],
                'ap': results['ap'],
                'best_accuracy': results['best_accuracy'],
                'best_threshold': results['best_threshold'],
                'real_count': folder_info['real_count'],
                'fake_count': folder_info['fake_count']
            }
            folder_result.update({f'acc_{t}': a for t, a in results['accuracies'].items()})
            all_results.append(folder_result)
            
        except Exception as e:
            print(f"Error processing folder: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        excel_path = os.path.join(result_folder, 'validation_results.xlsx')
        results_df.to_excel(excel_path, index=False)
        
        # Calculate averages
        avg_ap = results_df['ap'].mean()
        avg_acc = results_df['best_accuracy'].mean()
        
        print(f"\n📊 Overall Results:")
        print(f"   Average AP: {avg_ap:.4f}")
        print(f"   Average Best Accuracy: {avg_acc:.4f}")
        print(f"   Results saved to: {excel_path}")
        
        # Print summary table
        print(f"\n📋 Summary Table:")
        print(results_df[['folder', 'ap', 'best_accuracy', 'best_threshold']].to_string(index=False))
        
    else:
        print("❌ No successful validations completed!")

    print(f"\n🎉 Validation completed!")



if __name__ == '__main__':
    main()