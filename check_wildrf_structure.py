#!/usr/bin/env python3
"""
Check WildRF dataset folder structure and image counts for training/validation/test splits.
"""
import os
import glob

def count_images(folder):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    count = 0
    for file in os.listdir(folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            count += 1
    return count

def check_split(base_path, split):
    split_path = os.path.join(base_path, split)
    print(f"\nChecking {split} split: {split_path}")
    if not os.path.exists(split_path):
        print(f"  ❌ Split folder does not exist!")
        return
    generator_folders = [d for d in glob.glob(os.path.join(split_path, '*')) if os.path.isdir(d)]
    if not generator_folders:
        print(f"  ❌ No generator folders found!")
        return
    for gen_folder in generator_folders:
        print(f"- Generator: {os.path.basename(gen_folder)}")
        real_candidates = glob.glob(os.path.join(gen_folder, '*real*')) + glob.glob(os.path.join(gen_folder, '0_*'))
        fake_candidates = glob.glob(os.path.join(gen_folder, '*fake*')) + glob.glob(os.path.join(gen_folder, '1_*'))
        if not real_candidates:
            print("    ⚠️ No real folder found!")
        for real_cand in real_candidates:
            if os.path.isdir(real_cand):
                n = count_images(real_cand)
                print(f"    Real: {os.path.basename(real_cand)} - {n} images")
        if not fake_candidates:
            print("    ⚠️ No fake folder found!")
        for fake_cand in fake_candidates:
            if os.path.isdir(fake_cand):
                n = count_images(fake_cand)
                print(f"    Fake: {os.path.basename(fake_cand)} - {n} images")

def main():
    wildrf_base = "/mnt/mmlab2024nas/danh/phatlh/D3/data/WildRF"
    for split in ['train', 'val', 'test']:
        check_split(wildrf_base, split)

if __name__ == "__main__":
    main()
