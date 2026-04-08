
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pickle

device = "cpu"  # Force CPU
print(f"Running on {device}")

model_id = "openai/clip-vit-base-patch32" # this is CLIP model I used but you can change it
image_folder = r"path/directory"
output_file = "image_features.pkl"  # this is the feature output file


def extract_clip_features():
    print(f"Loading CLIP model ({model_id})...") # loading the above model
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    if not os.path.exists(image_folder): # loading theimages to extract features 
        print(f"Folder not found {image_folder}")
        return

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] # add/remove extensions as needed
    if not image_files:
        print("No images found - Check path")
        return

    print(f"Found {len(image_files)} images. Processing...")

    features_dict = {}

    for i, img_file in enumerate(image_files): # processes the images in the folder
        img_path = os.path.join(image_folder, img_file)
        
        try:
            image = Image.open(img_path).convert("RGB") # this converts evertyhing to RGB just incase there are greyscale images
            
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                image_features = model.get_image_features(**inputs) # extracts features from the image using CLIP

            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, 'image_embeds'):
                    image_features = image_features.image_embeds
                elif hasattr(image_features, 'pooler_output'):
                    image_features = image_features.pooler_output
                else:
                    image_features = image_features[0]

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True) # normalizes the features to unit length

            features_dict[img_file] = image_features.cpu().numpy() #numpy storees features as a value

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(image_files)}")

        except Exception as e:
            print(f"Problem reading {img_file}: {e}")

    with open(output_file, 'wb') as f: # saves the features to a pickle file (which we need in the next step)
        pickle.dump(features_dict, f)
    
    print(f"All done. The features are saved to {output_file}")

if __name__ == "__main__":
    extract_clip_features()
