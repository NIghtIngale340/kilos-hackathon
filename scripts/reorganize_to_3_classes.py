#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reorganize pole dataset into 3 clean classes: Urgent, Moderate, Normal
Ensures all 3 classes exist in train/valid/test splits
"""

import os
import shutil
from pathlib import Path

# Mapping from original 7 classes to 3 classes
CLASS_MAPPING = {
    # URGENT - Immediate safety hazards
    'Old Pole': 'urgent',
    'Urgent': 'urgent',
    
    # MODERATE - Vegetation hazards (should be fixed soon)
    'Vegetation Pole': 'moderate',
    'Pole Vegetation': 'moderate',
    'UnlabeledPole Vegetation': 'moderate',
    
    # NORMAL - No immediate issues
    'Normal': 'normal',
    'Unlabeled': 'normal',  # Assuming unlabeled means no visible issues
}

def reorganize_dataset(source_path, dest_path):
    """Reorganize dataset into 3 classes"""
    
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    
    if dest_path.exists():
        print(f"⚠️  Destination '{dest_path}' already exists!")
        response = input("Delete and recreate? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
        shutil.rmtree(dest_path)
    
    print(f"📁 Source: {source_path}")
    print(f"📁 Destination: {dest_path}")
    print(f"\n🗂️  Class mapping:")
    for old_class, new_class in CLASS_MAPPING.items():
        print(f"  {old_class} → {new_class}")
    print()
    
    # Create destination directories
    for split in ['train', 'valid', 'test']:
        for new_class in ['urgent', 'moderate', 'normal']:
            (dest_path / split / new_class).mkdir(parents=True, exist_ok=True)
    
    # Copy files
    total_files = 0
    stats = {'train': {}, 'valid': {}, 'test': {}}
    
    for split in ['train', 'valid', 'test']:
        split_dir = source_path / split
        if not split_dir.exists():
            print(f"⚠️  {split} directory not found, skipping...")
            continue
            
        print(f"\n📂 Processing {split}/")
        
        for old_class_dir in split_dir.iterdir():
            if not old_class_dir.is_dir():
                continue
                
            old_class = old_class_dir.name
            
            # Skip if class not in mapping
            if old_class not in CLASS_MAPPING:
                print(f"  ⚠️  Skipping unknown class: {old_class}")
                continue
            
            new_class = CLASS_MAPPING[old_class]
            
            # Copy all images
            image_count = 0
            for img_file in old_class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Create unique filename
                    new_name = f"{old_class.replace(' ', '_')}_{img_file.name}"
                    dest_file = dest_path / split / new_class / new_name
                    shutil.copy2(img_file, dest_file)
                    image_count += 1
                    total_files += 1
            
            print(f"  ✓ {old_class} → {new_class}: {image_count} images")
            
            # Update stats
            if new_class not in stats[split]:
                stats[split][new_class] = 0
            stats[split][new_class] += image_count
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 REORGANIZATION SUMMARY")
    print("=" * 70)
    
    for split in ['train', 'valid', 'test']:
        print(f"\n{split.upper()}:")
        for class_name in ['urgent', 'moderate', 'normal']:
            count = stats[split].get(class_name, 0)
            print(f"  {class_name:12s}: {count:3d} images")
    
    print(f"\n✅ Total files copied: {total_files}")
    print(f"📁 New dataset location: {dest_path}")
    
    # Check if all classes exist in all splits
    print("\n" + "=" * 70)
    print("✓ VALIDATION")
    print("=" * 70)
    
    all_good = True
    for split in ['train', 'valid', 'test']:
        missing = []
        for class_name in ['urgent', 'moderate', 'normal']:
            if stats[split].get(class_name, 0) == 0:
                missing.append(class_name)
        
        if missing:
            print(f"⚠️  {split}: Missing classes: {', '.join(missing)}")
            all_good = False
        else:
            print(f"✅ {split}: All 3 classes present")
    
    if all_good:
        print("\n🎉 Dataset is ready for training!")
    else:
        print("\n⚠️  Warning: Some splits are missing classes. Training may fail.")
        print("   Consider manually adding images to missing classes.")

if __name__ == '__main__':
    source = Path(__file__).parent.parent / 'pole-mock-dataset'
    dest = Path(__file__).parent / 'pole-3class-dataset'
    
    reorganize_dataset(source, dest)
