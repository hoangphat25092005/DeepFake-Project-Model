# Fine-tuning D³ Model on WildRF Dataset

## Overview
This guide shows you how to fine-tune the pre-trained D³ model on your custom WildRF dataset.

## Dataset Structure
Your WildRF dataset should follow this structure:
```
/mnt/mmlab2024nas/danh/phatlh/D3/data/WildRF/
├── train/
│   ├── facebook/
│   │   ├── 0_real/
│   │   │   ├── 0000.jpg
│   │   │   ├── 0001.jpg
│   │   │   └── ...
│   │   └── 1_fake/
│   │       ├── 0000.jpg
│   │       ├── 0001.jpg
│   │       └── ...
│   ├── stylegan2/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   └── ... (other generators)
├── val/
│   └── (same structure as train)
└── test/
    └── (same structure as train)
```

## Quick Start

### 1. Prepare Your Dataset Paths
First, identify all your training and validation data folders:

**Training Data:**
- Real images: `/mnt/mmlab2024nas/danh/phatlh/D3/data/WildRF/train/*/0_real`
- Fake images: `/mnt/mmlab2024nas/danh/phatlh/D3/data/WildRF/train/*/1_fake`

**Validation Data:**
- `/mnt/mmlab2024nas/danh/phatlh/D3/data/WildRF/val/facebook`
- `/mnt/mmlab2024nas/danh/phatlh/D3/data/WildRF/val/stylegan2`
- ... (other validation folders)

### 2. Use the Fine-tuning Script

I've created a ready-to-use script `finetune_wildrf.py` that handles everything automatically.

**Basic Usage:**
```bash
# Fine-tune from pre-trained checkpoint
CUDA_VISIBLE_DEVICES=0 python finetune_wildrf.py \
    --name=finetune_wildrf \
    --resume_path=/mnt/mmlab2024nas/danh/phatlh/D3/ckpt/classifier.pth \
    --niter=50 \
    --batch_size=32 \
    --lr=0.00001
```

**Advanced Options:**
```bash
CUDA_VISIBLE_DEVICES=0,1 python finetune_wildrf.py \
    --name=finetune_wildrf_advanced \
    --resume_path=/mnt/mmlab2024nas/danh/phatlh/D3/ckpt/classifier.pth \
    --niter=100 \
    --batch_size=64 \
    --lr=0.00001 \
    --earlystop_epoch=10 \
    --save_epoch_freq=5 \
    --data_aug
```

### 3. Training Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--name` | `finetune_wildrf` | Experiment name (creates folder in checkpoints/) |
| `--resume_path` | None | Path to pre-trained checkpoint (classifier.pth) |
| `--niter` | 50 | Number of training epochs |
| `--batch_size` | 32 | Batch size (reduce if OOM) |
| `--lr` | 0.00001 | Learning rate (lower for fine-tuning) |
| `--earlystop_epoch` | 5 | Early stopping patience |
| `--save_epoch_freq` | 1 | Save checkpoint every N epochs |
| `--data_aug` | False | Enable data augmentation (blur, JPEG) |
| `--fix_backbone` | True | Freeze CLIP backbone (recommended) |

### 4. Monitor Training

Training logs and checkpoints will be saved in:
```
./checkpoints/finetune_wildrf/
├── train/              # TensorBoard training logs
├── val/                # TensorBoard validation logs
├── results.xlsx        # Validation metrics per epoch
├── model_epoch_0.pth   # Checkpoints every epoch
├── model_epoch_5.pth
└── model_epoch_best.pth  # Best model
```

**View TensorBoard:**
```bash
tensorboard --logdir=./checkpoints/finetune_wildrf
```

### 5. Training Strategies

#### Strategy 1: Quick Fine-tuning (Recommended)
For adapting to WildRF dataset quickly:
```bash
CUDA_VISIBLE_DEVICES=0 python finetune_wildrf.py \
    --name=quick_finetune \
    --resume_path=./ckpt/classifier.pth \
    --niter=30 \
    --batch_size=32 \
    --lr=0.00001 \
    --fix_backbone
```
- ✅ Freezes CLIP backbone (faster training)
- ✅ Only trains attention head
- ✅ Takes ~2-4 hours

#### Strategy 2: Full Fine-tuning
For maximum performance:
```bash
CUDA_VISIBLE_DEVICES=0,1 python finetune_wildrf.py \
    --name=full_finetune \
    --resume_path=./ckpt/classifier.pth \
    --niter=100 \
    --batch_size=64 \
    --lr=0.000001 \
    --data_aug
```
- ⚠️ Trains entire model (slower)
- ⚠️ Requires more GPU memory
- ⚠️ Takes ~8-12 hours

#### Strategy 3: From Scratch (Not Recommended)
Only if you have very different data:
```bash
CUDA_VISIBLE_DEVICES=0,1 python finetune_wildrf.py \
    --name=train_from_scratch \
    --niter=200 \
    --batch_size=128 \
    --lr=0.0001
```

### 6. Validation and Testing

After training, evaluate your fine-tuned model:

```bash
# Update validate_for_robustness.py checkpoint path
sed -i 's|ckpt/classifier.pth|checkpoints/finetune_wildrf/model_epoch_best.pth|g' validate_for_robustness.py

# Run validation
CUDA_VISIBLE_DEVICES=0 python validate_for_robustness.py
```

Or use the visualization script:
```bash
# Update visualize_result.py checkpoint path
CUDA_VISIBLE_DEVICES=0 python visualize_result.py
```

### 7. Common Issues & Solutions

#### Issue 1: CUDA Out of Memory
```bash
# Solution: Reduce batch size
python finetune_wildrf.py --batch_size=16
```

#### Issue 2: Training Too Slow
```bash
# Solution: Reduce workers or use smaller images
python finetune_wildrf.py --num_threads=2 --batch_size=32
```

#### Issue 3: Model Not Improving
```bash
# Solution: Adjust learning rate
python finetune_wildrf.py --lr=0.00005  # Higher LR
# Or enable data augmentation
python finetune_wildrf.py --data_aug
```

## Expected Results

After fine-tuning on WildRF, you should see:
- **ID Accuracy**: 95-98% (on WildRF generators in training)
- **OOD Accuracy**: 85-90% (on unseen generators)
- **Training Time**: 2-6 hours (depends on dataset size and GPU)

## Advanced Configuration

### Custom Data Augmentation
Edit the augmentation parameters in the script:
```python
--blur_prob=0.5      # 50% chance to apply Gaussian blur
--blur_sig=0.0,3.0   # Blur sigma range
--jpg_prob=0.5       # 50% chance to apply JPEG compression
--jpg_qual=30,100    # JPEG quality range
```

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_wildrf.py \
    --gpu_ids=0,1,2,3 \
    --batch_size=256
```

### Resume from Interrupted Training
```bash
python finetune_wildrf.py \
    --name=finetune_wildrf \
    --resume_path=./checkpoints/finetune_wildrf/model_epoch_10.pth
```

## Tips for Best Performance

1. **Start with pre-trained weights** (classifier.pth) - Much better than training from scratch
2. **Use smaller learning rate** (0.00001) for fine-tuning
3. **Enable early stopping** to prevent overfitting
4. **Monitor both train and val losses** in TensorBoard
5. **Save checkpoints frequently** in case training crashes
6. **Test on multiple generators** to verify generalization

## Questions?

If you encounter any issues:
1. Check the training logs in `checkpoints/[name]/train/`
2. Verify your dataset structure matches the expected format
3. Try reducing batch size if you get OOM errors
4. Enable data augmentation if model overfits quickly
