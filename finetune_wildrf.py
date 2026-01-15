#!/usr/bin/env python3
"""
Fine-tuning script for D³ model on WildRF dataset
Automatically discovers training/validation data from WildRF folder structure
"""

import os
import glob
import numpy as np
import pandas as pd
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
import torch


def discover_wildrf_data(base_path, split='train'):
    """
    Automatically discover real and fake folders in WildRF dataset
    Args:
        base_path: Path to WildRF folder (e.g., /path/to/WildRF)
        split: 'train', 'val', or 'test'
    Returns:
        real_folders: List of paths to real image folders
        fake_folders: List of paths to fake image folders
    """
    split_path = os.path.join(base_path, split)
    
    if not os.path.exists(split_path):
        print(f" Warning: {split_path} does not exist!")
        return [], []
    
    real_folders = []
    fake_folders = []
    
    # Search for generator folders (e.g., facebook, stylegan2, etc.)
    generator_folders = [d for d in glob.glob(os.path.join(split_path, '*')) 
                        if os.path.isdir(d)]
    
    print(f"\n Discovering {split} data in: {split_path}")
    print(f"   Found {len(generator_folders)} generator folders")
    
    for gen_folder in generator_folders:
        gen_name = os.path.basename(gen_folder)
        
        # Look for real folders (0_real, real, Real, etc.)
        real_candidates = glob.glob(os.path.join(gen_folder, '*real*'))
        real_candidates += glob.glob(os.path.join(gen_folder, '0_*'))
        
        # Look for fake folders (1_fake, fake, Fake, etc.)
        fake_candidates = glob.glob(os.path.join(gen_folder, '*fake*'))
        fake_candidates += glob.glob(os.path.join(gen_folder, '1_*'))
        
        # Verify folders contain images
        for real_cand in real_candidates:
            if os.path.isdir(real_cand):
                image_count = count_images(real_cand)
                if image_count > 0:
                    real_folders.append(real_cand)
                    print(f" {gen_name}/real: {image_count} images")
        
        for fake_cand in fake_candidates:
            if os.path.isdir(fake_cand):
                image_count = count_images(fake_cand)
                if image_count > 0:
                    fake_folders.append(fake_cand)
                    print(f"  {gen_name}/fake: {image_count} images")
    
    print(f"\n Summary for {split} split:")
    print(f"   Real folders: {len(real_folders)}")
    print(f"   Fake folders: {len(fake_folders)}")
    
    return real_folders, fake_folders


def count_images(folder):
    """Count number of images in a folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    count = 0
    try:
        for file in os.listdir(folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                count += 1
    except:
        pass
    return count


def discover_validation_folders(base_path):
    """
    Discover all validation generator folders
    Returns a list of paths like ['.../val/facebook', '.../val/stylegan2', ...]
    """
    val_path = os.path.join(base_path, 'val')
    
    if not os.path.exists(val_path):
        print(f"Warning: {val_path} does not exist!")
        return []
    
    generator_folders = [d for d in glob.glob(os.path.join(val_path, '*')) 
                        if os.path.isdir(d)]
    
    valid_folders = []
    for gen_folder in generator_folders:
        # Check if it has real/fake subfolders with images
        real_folders = glob.glob(os.path.join(gen_folder, '*real*'))
        fake_folders = glob.glob(os.path.join(gen_folder, '*fake*'))
        
        if real_folders and fake_folders:
            real_count = count_images(real_folders[0])
            fake_count = count_images(fake_folders[0])
            if real_count > 0 and fake_count > 0:
                valid_folders.append(gen_folder)
                print(f"   Validation: {os.path.basename(gen_folder)} (real: {real_count}, fake: {fake_count})")
    
    return valid_folders


def get_val_opt(train_opt):
    """Create validation options from training options"""
    val_opt = TrainOptions().parse(print_options=False)
    
    # Copy settings from train_opt
    for key, value in vars(train_opt).items():
        setattr(val_opt, key, value)
    
    # Override specific settings for validation
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


def main():
    """Main fine-tuning function"""
    print("\n" + "="*80)
    print("D³ MODEL FINE-TUNING ON WILDRF DATASET")
    print("="*80)
    
    # Set random seed for reproducibility
    seed = 418
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Parse training options
    opt = TrainOptions().parse()
    
    # WildRF dataset base path
    wildrf_base = "/mnt/mmlab2024nas/danh/phatlh/D3/data/WildRF"
    
    # Discover training data
    print("\n" + "="*80)
    print("DISCOVERING TRAINING DATA")
    print("="*80)
    real_folders, fake_folders = discover_wildrf_data(wildrf_base, split='train')
    
    if not real_folders or not fake_folders:
        print("\n ERROR: No training data found!")
        print("   Please ensure your dataset follows this structure:")
        print("   WildRF/train/0_real/")
        print("   WildRF/train/1_fake/")
        print("   (and similarly for val and test splits)")
        return
    
    # Create training data loader
    print("\nCreating training data loader...")
    data_loader = create_dataloader(opt, real_folders, fake_folders)
    print(f"Training data loader created: {len(data_loader)} batches")
    
    # Initialize model
    print("\nInitializing model...")
    model = Trainer(opt)
    print("Model initialized")
    
    # Discover validation data
    print("\n" + "="*80)
    print("DISCOVERING VALIDATION DATA")
    print("="*80)
    val_data_roots = discover_validation_folders(wildrf_base)
    
    if not val_data_roots:
        print("Warning: No validation data found! Training without validation.")
    
    # Create validation options and data loaders
    val_opt = get_val_opt(opt)
    val_loader_list = []
    
    for root in val_data_roots:
        real_folders_val = glob.glob(os.path.join(root, '*real*'))
        fake_folders_val = glob.glob(os.path.join(root, '*fake*'))
        
        if real_folders_val and fake_folders_val:
            val_loader_list.append(
                create_dataloader(val_opt, real_folders_val, fake_folders_val)
            )
    
    print(f"\nCreated {len(val_loader_list)} validation data loaders")
    
    # Create output directories
    checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize TensorBoard writers
    train_writer = SummaryWriter(os.path.join(checkpoint_dir, "train"))
    val_writer = SummaryWriter(os.path.join(checkpoint_dir, "val"))
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Experiment name: {opt.name}")
    print(f"Total epochs: {opt.niter}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Learning rate: {opt.lr}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("="*80 + "\n")
    
    start_time = time.time()
    results_dict = {}
    
    for epoch in range(opt.niter):
        epoch_start = time.time()
        print(f"\n{'='*80}")
        print(f'EPOCH {epoch+1}/{opt.niter}')
        print(f"{'='*80}")
        
        # Training
        model.train()
        epoch_losses = []
        
        for i, data in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch+1}")):
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()
            epoch_losses.append(model.loss)
            
            if model.total_steps % opt.loss_freq == 0:
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                avg_time = (time.time() - start_time) / model.total_steps
                print(f"\n   Step {model.total_steps}: Loss = {model.loss:.4f}, Time/iter = {avg_time:.3f}s")
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        print(f"\n   Epoch {epoch+1} completed in {epoch_time/60:.2f} minutes")
        print(f"   Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if epoch % opt.save_epoch_freq == 0:
            print(f'\nSaving checkpoint at epoch {epoch}')
            model.save_networks('model_epoch_best.pth')
            model.save_networks(f'model_epoch_{epoch}.pth')
        
        # Validation
        if val_loader_list:
            print(f"\nRunning validation...")
            model.eval()
            acc_list = []
            ap_list = []
            b_acc_list = []
            threshold_list = []
            y_pred_list = []
            y_true_list = []
            
            for i, val_loader in enumerate(val_loader_list):
                val_name = os.path.basename(val_data_roots[i])
                ap, r_acc0, f_acc0, acc, r_acc1, f_acc1, acc1, best_thres, y_pred, y_true = validate(
                    model.model, val_loader, find_thres=True
                )
                acc_list.append(acc)
                ap_list.append(ap)
                b_acc_list.append(acc1)
                threshold_list.append(best_thres)
                
                val_writer.add_scalar(f'accuracy_{val_name}', acc, model.total_steps)
                val_writer.add_scalar(f'ap_{val_name}', ap, model.total_steps)
                
                print(f"   {val_name}: AP={ap:.4f}, Acc={acc:.4f}, R_Acc={r_acc1:.4f}, F_Acc={f_acc1:.4f}")
                
                y_pred_list.append(y_pred)
                y_true_list.append(y_true)
            
            # Average metrics
            avg_ap = sum(ap_list) / len(val_loader_list)
            avg_acc = sum(acc_list) / len(val_loader_list)
            avg_b_acc = sum(b_acc_list) / len(val_loader_list)
            avg_threshold = sum(threshold_list) / len(val_loader_list)
            
            print(f"\nAverage Validation: AP={avg_ap:.4f}, Acc={avg_acc:.4f}")
            
            # Save results
            ap_list.append(avg_ap)
            acc_list.append(avg_acc)
            b_acc_list.append(avg_b_acc)
            threshold_list.append(avg_threshold)
            
            results_dict[f'epoch_{epoch}_ap'] = ap_list
            results_dict[f'epoch_{epoch}_acc'] = acc_list
            results_dict[f'epoch_{epoch}_b_acc'] = b_acc_list
            results_dict[f'epoch_{epoch}_threshold'] = threshold_list
            
            results_df = pd.DataFrame(results_dict)
            results_df.to_excel(os.path.join(checkpoint_dir, 'results.xlsx'), 
                              sheet_name='sheet1', index=False)
            
            # Save predictions
            np.savez(os.path.join(checkpoint_dir, f'y_pred_eval_{epoch}.npz'), *y_pred_list)
            np.savez(os.path.join(checkpoint_dir, f'y_true_eval_{epoch}.npz'), *y_true_list)
            
            # Early stopping check
            early_stopping(avg_acc, model)
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("\nLearning rate dropped by 10x, continuing training...")
                    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    print("\nEarly stopping triggered. Training finished.")
                    break
    
    # Training completed
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"Best model: {os.path.join(checkpoint_dir, 'model_epoch_best.pth')}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
