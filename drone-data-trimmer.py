import os
import random
import shutil

# Paths
train_img_path = './drone_data/train/images'
train_lbl_path = './drone_data/train/labels'
backup_path = './drone_data/train_backup'

# Target number of images for ~15 min epochs
TARGET_COUNT = 10000 

os.makedirs(backup_path, exist_ok=True)
os.makedirs(os.path.join(backup_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(backup_path, 'labels'), exist_ok=True)

# Get list of all images
all_images = [f for f in os.listdir(train_img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
total_current = len(all_images)

if total_current <= TARGET_COUNT:
    print(f"Dataset already small enough ({total_current} images). No shrinking needed.")
else:
    # Select which images to REMOVE
    to_remove_count = total_current - TARGET_COUNT
    to_remove = random.sample(all_images, to_remove_count)
    
    print(f"Shrinking dataset from {total_current} to {TARGET_COUNT}...")
    
    for img_name in to_remove:
        # Move Image
        shutil.move(os.path.join(train_img_path, img_name), 
                    os.path.join(backup_path, 'images', img_name))
        
        # Move corresponding Label
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_full_path = os.path.join(train_lbl_path, lbl_name)
        if os.path.exists(lbl_full_path):
            shutil.move(lbl_full_path, os.path.join(backup_path, 'labels', lbl_name))

    print("Cleanup complete. Ready to restart training!")