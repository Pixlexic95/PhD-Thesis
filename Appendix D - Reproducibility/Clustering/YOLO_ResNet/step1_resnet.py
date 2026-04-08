
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import normalize # <--- ADDED IMPORT
from PIL import Image
import os
import glob
import numpy as np

def extract_resnet_features():
    input_folder = "/path/directory/"
    output_file = "features_resnet50.npy"# change depending on model 
    filename_file = "filenames.npy"

    print("Loading ResNet model...")
    weights = models.ResNet50_Weights.IMAGENET1K_V1 # ours is pretrained on ImageNet
    model = models.resnet50(weights=weights)
    
    modules = list(model.children())[:-1] #remove the final (fc) layer
    model = nn.Sequential(*modules)
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    preprocess = weights.transforms() #preprocessing

    print(f"Scanning {input_folder}...")
    unique_paths = set() # prevents duplicates
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.PNG', '*.JPEG']# We scan for all patterns, but the set will ignore repeats
    
    for ext in extensions:
        found = glob.glob(os.path.join(input_folder, "**", ext), recursive=True)
        for f in found:
            clean_path = os.path.normpath(f)# Normalize path separators to ensure uniqueness
            unique_paths.add(clean_path)
    
    image_paths = sorted(list(unique_paths))# Convert back to a sorted list so the order is permanent
    print(f"Found {len(image_paths)} images.")

    if len(image_paths) == 0:
        print("No images found.")
        return

    features_list = [] #extraction loop
    valid_paths = []
    print("Starting extraction...")
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = preprocess(img).unsqueeze(0).to(device)
                
                embedding = model(img_t)
                feat = embedding.flatten().cpu().numpy()
                
                features_list.append(feat)
                valid_paths.append(img_path)
                
            except Exception as e:
                print(f"Problem reading {os.path.basename(img_path)}: {e}")

            if (i+1) % 50 == 0:
                print(f"  Processed {i+1}/{len(image_paths)}...")

    features = np.array(features_list) #normalize and save
    
    print("-" * 30)
    print(f"Raw features shape: {features.shape}")
    
    print("Applying L2 Normalization (Fairness for Classification)...") #L2 normalization
    features = normalize(features, norm='l2', axis=1) # norm='l2' means the vector length becomes 1.0. removes brightness magnitude so we only cluster on shape/texture.
    
    np.save(output_file, features)
    np.save(filename_file, np.array(valid_paths))
    
    print("-" * 30)
    print(f"All done")
    print(f"Saved unique, NORMALIZED features to: {output_file}")
    print(f"Saved unique filenames to: {filename_file}")
    print("-" * 30)

if __name__ == '__main__':
    extract_resnet_features()
