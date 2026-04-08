
import torch
from ultralytics import YOLO
from sklearn.preprocessing import normalize
import os
import glob
import numpy as np

def extract_yolo_features():
    input_folder = "/path/directory/"
    output_file = "features_test.npy"
    filename_file = "filenames_test.npy"

    model_name = 'yolo11x-cls.pt'# yolo11x-cls.pt is the strongest (comparable to ResNet50/101) yolo11n-cls.pt is the fastest.
    print(f"Loading {model_name}...")
    
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Problem loading model: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check gpu
    print(f"Model loaded. Running on: {device}")

    print(f"Checking {input_folder}")
    unique_paths = set()
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.PNG', '*.JPEG']
    
    for ext in extensions:
        found = glob.glob(os.path.join(input_folder, "**", ext), recursive=True)
        for f in found:
            clean_path = os.path.normpath(f)
            unique_paths.add(clean_path)
    
    image_paths = sorted(list(unique_paths))
    print(f"Found {len(image_paths)} unique images.")

    if len(image_paths) == 0:
        print("No images found")
        return

    features_list = []
    valid_paths = []

    print("Starting extraction (using model.embed)...")

    for i, img_path in enumerate(image_paths):
        try:
            embedding = model.embed(img_path, verbose=False, device=device) # model.embed() extracts features from the penultimate layer. It handles loading, resizing, and normalizing internally.
            feat = embedding[0].cpu().numpy()  # The result is a tensor, we need to move it to CPU numpyEmbedding shape is usually (1, 1280) for large models
            
            features_list.append(feat)
            valid_paths.append(img_path)
            
        except Exception as e:
            print(f"Problem reading {os.path.basename(img_path)}: {e}")

        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(image_paths)}...")

    features = np.array(features_list) #normalize
    
    print("-" * 30)
    print(f"Raw features shape: {features.shape}")
    
    print("Applying L2 Normalization")
    features = normalize(features, norm='l2', axis=1)# This aligns YOLO geometry with the others
    
    np.save(output_file, features)
    np.save(filename_file, np.array(valid_paths))
    
    print("-" * 30)
    print(f"all done")
    print(f"Saved NORMALIZED features to: {output_file}")
    print(f"Saved filenames to: {filename_file}")
    print("-" * 30)

if __name__ == '__main__':
    extract_yolo_features()
