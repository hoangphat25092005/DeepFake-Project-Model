import os
import numpy as np
import pandas as pd
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
import torch


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
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


if __name__ == '__main__':
    seed = 418
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    opt = TrainOptions().parse()
    
    # Check if resume path exists
    if opt.resume_path and not os.path.exists(opt.resume_path):
        print(f"\nERROR: Resume checkpoint not found: {opt.resume_path}")
        print("Please check the path and try again.")
        exit(1)
    
    val_opt = get_val_opt()
    
    # WildRF dataset paths
    wildrf_base = "/mnt/mmlab2024nas/danh/phatlh/D3/data/WildRF"
    
    # Training data - directly use 0_real and 1_fake folders
    real_folders = [os.path.join(wildrf_base, "train", "0_real")]
    fake_folders = [os.path.join(wildrf_base, "train", "1_fake")]
    
    # Check if training folders exist
    if not os.path.exists(real_folders[0]) or not os.path.exists(fake_folders[0]):
        print(f"\nERROR: Training data folders not found!")
        print(f"Expected:")
        print(f"  Real: {real_folders[0]}")
        print(f"  Fake: {fake_folders[0]}")
        print(f"\nPlease organize your data as described earlier.")
        exit(1)
    
    print(f"\n{'='*80}")
    print("TRAINING DATA")
    print(f"{'='*80}")
    print(f"Real folder: {real_folders[0]}")
    print(f"Fake folder: {fake_folders[0]}")
    
    # Count images
    from glob import glob
    real_images = len(glob(os.path.join(real_folders[0], "*.jpg"))) + len(glob(os.path.join(real_folders[0], "*.png")))
    fake_images = len(glob(os.path.join(fake_folders[0], "*.jpg"))) + len(glob(os.path.join(fake_folders[0], "*.png")))
    print(f"Real images: {real_images}")
    print(f"Fake images: {fake_images}")
    print(f"{'='*80}\n")
    
    if real_images == 0 or fake_images == 0:
        print("ERROR: No images found in training folders!")
        exit(1)
    
    data_loader = create_dataloader(opt, real_folders, fake_folders)

    # Initialize detector
    print(f"\n{'='*80}")
    print("INITIALIZING MODEL")
    print(f"{'='*80}")
    print(f"Architecture: {opt.arch}")
    if opt.resume_path:
        print(f"Loading checkpoint: {opt.resume_path}")
    model = Trainer(opt)
    
    if model.model is None:
        print("\nERROR: Model initialization failed!")
        print("This usually means:")
        print("  1. The architecture name is incorrect")
        print("  2. The checkpoint file is corrupted")
        print("  3. There's a mismatch between checkpoint and architecture")
        exit(1)
    
    print("Model initialized successfully")
    print(f"{'='*80}\n")

    # Validation data - test split with generator folders
    val_data_root = [
        os.path.join(wildrf_base, "test", "facebook"),
        os.path.join(wildrf_base, "test", "reddit"),
        os.path.join(wildrf_base, "test", "twitter")
    ]
    
    # Filter only existing validation folders
    val_data_root = [root for root in val_data_root if os.path.exists(root)]
    print(f"\n{'='*80}")
    print("VALIDATION DATA")
    print(f"{'='*80}")
    print(f"Found {len(val_data_root)} validation generators")
    for root in val_data_root:
        print(f"  - {os.path.basename(root)}")

    # Initialize val datasets 
    val_loader_list = []
    for root in val_data_root:
        # Look for real and fake subfolders (0_real, 1_fake, real, fake, etc.)
        real_candidates = [
            os.path.join(root, "0_real"),
            os.path.join(root, "real")
        ]
        fake_candidates = [
            os.path.join(root, "1_fake"),
            os.path.join(root, "fake")
        ]
        
        # Use first existing folder
        real_folder = next((f for f in real_candidates if os.path.exists(f)), None)
        fake_folder = next((f for f in fake_candidates if os.path.exists(f)), None)
        
        if real_folder and fake_folder:
            val_loader_list.append(
                create_dataloader(val_opt, [real_folder], [fake_folder])
            )
            print(f"Added validation loader for {os.path.basename(root)}")
        else:
            print(f"Skipped {os.path.basename(root)} (missing real/fake folders)")

    print(f"\nTotal validation loaders: {len(val_loader_list)}")
    print(f"{'='*80}\n")

    # Create output directories
    checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_writer = SummaryWriter(os.path.join(checkpoint_dir, "train"))
    val_writer = SummaryWriter(os.path.join(checkpoint_dir, "val"))
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Experiment name: {opt.name}")
    print(f"Training batches: {len(data_loader)}")
    print(f"Total epochs: {opt.niter}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Learning rate: {opt.lr}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Save frequency: Every {opt.save_epoch_freq} epochs")
    print(f"Early stopping patience: {opt.earlystop_epoch} epochs")
    print(f"{'='*80}\n")
    
    results_dict = {}
    for epoch in range(opt.niter):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{opt.niter}")
        print(f"{'='*80}")
        
        # Training
        model.train()
        epoch_losses = []
        
        for i, data in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch+1}")):
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()
            epoch_losses.append(model.loss.item())  # Convert to Python float

            if model.total_steps % opt.loss_freq == 0:
                train_writer.add_scalar('loss', model.loss.item(), model.total_steps)
                avg_time = (time.time()-start_time)/model.total_steps
                print(f"\n  Step {model.total_steps}: Loss={model.loss.item():.4f}, Time/iter={avg_time:.3f}s")

        # Epoch summary
        epoch_time = time.time() - start_time
        avg_loss = np.mean(epoch_losses)
        print(f"\n  Epoch completed in {(epoch_time)/(epoch+1)/60:.2f} min/epoch")
        print(f"  Average loss: {avg_loss:.4f}")

        # Save checkpoint
        if epoch % opt.save_epoch_freq == 0:
            print(f'\nSaving checkpoint at epoch {epoch}')
            model.save_networks('model_epoch_best.pth')
            model.save_networks(f'model_epoch_{epoch}.pth')

        # Validation
        if val_loader_list:
            print(f"\n  Running validation...")
            model.eval()
            acc_list = []
            ap_list = []
            b_acc_list = []
            threshold_list = []
            y_pred_list = []
            y_true_list = []
            
            for i, val_loader in enumerate(val_loader_list):
                val_name = os.path.basename(val_data_root[i])
                ap, r_acc0, f_acc0, acc, r_acc1, f_acc1, acc1, best_thres, y_pred, y_true = validate(
                    model.model, val_loader, find_thres=True
                )
                acc_list.append(acc)
                ap_list.append(ap)
                b_acc_list.append(acc1)
                threshold_list.append(best_thres)
                
                val_writer.add_scalar(f'accuracy_{val_name}', acc, model.total_steps)
                val_writer.add_scalar(f'ap_{val_name}', ap, model.total_steps)
                
                print(f"{val_name}: AP={ap:.4f}, Acc={acc:.4f}, "
                      f"R_Acc={r_acc1:.4f}, F_Acc={f_acc1:.4f}")
                
                y_pred_list.append(y_pred)
                y_true_list.append(y_true)

            # Average metrics
            avg_ap = sum(ap_list) / len(val_loader_list)
            avg_acc = sum(acc_list) / len(val_loader_list)
            avg_b_acc = sum(b_acc_list) / len(val_loader_list)
            avg_threshold = sum(threshold_list) / len(val_loader_list)
            
            ap_list.append(avg_ap)
            acc_list.append(avg_acc)
            b_acc_list.append(avg_b_acc)
            threshold_list.append(avg_threshold)
            
            results_dict[f'epoch_{epoch}_ap'] = ap_list
            results_dict[f'epoch_{epoch}_acc'] = acc_list
            results_dict[f'epoch_{epoch}_b_acc'] = b_acc_list
            results_dict[f'epoch_{epoch}_b_threshold'] = threshold_list
            
            results_df = pd.DataFrame(results_dict)
            results_df.to_excel(os.path.join(checkpoint_dir, 'results.xlsx'), 
                              sheet_name='sheet1', index=False)
            
            print(f"\nAverage Validation: AP={avg_ap:.4f}, Acc={avg_acc:.4f}")
            
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
                    print("\nEarly stopping triggered.")
                    break
        
        model.train()
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Best model: {os.path.join(checkpoint_dir, 'model_epoch_best.pth')}")
    print("="*80 + "\n")