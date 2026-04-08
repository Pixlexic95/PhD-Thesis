
import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

MODEL_PATH = r"C:\path"
INPUT_DIR = r"C:\path"
OUTPUT_DIR = r"C:\path"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class YOLOGeneralWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model

    def forward(self, x):
        results = self.model(x)
        # results[0] shape: [1, 4 + num_classes, num_anchors]
        
        # Get the class scores (skip first 4 rows which are boxes)
        class_scores = results[0][:, 4:, :]
        
        # Find the absolute maximum score in the entire prediction - returns a [1, 1] tensor
        max_score = torch.max(class_scores).view(1, 1)
        
        return max_score

def process_folder():
    print("Starting Grad-CAM")
    yolo_model = YOLO(MODEL_PATH) #opens model
    wrapped_model = YOLOGeneralWrapper(yolo_model)
    wrapped_model.train()
    for param in wrapped_model.parameters():
        param.requires_grad = True

    # Target convolutional blocks for YOLOv9m
    target_layers = [
        yolo_model.model.model[21], #RepNCSPELAN4
        yolo_model.model.model[18], # adjustable but chose a balance
        yolo_model.model.model[15]
    ]
    print("Targeting layer:", target_layers[0])
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    
    # Target the only output we have (index 0)
    targets = [ClassifierOutputTarget(0)]

    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(valid_extensions)])
    print(f"Found {len(image_files)} images. Processing.")

    for i, filename in enumerate(image_files):
        img_path = os.path.join(INPUT_DIR, filename)
        
        img = cv2.imread(img_path)
        if img is None: continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (640, 640))
        img_float = np.float32(img_res) / 255
        input_tensor = torch.from_numpy(np.transpose(img_float, (2, 0, 1))).unsqueeze(0).cuda()

        # Run Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Create and Save Visualization
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        final_output = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        
        file_stem = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{file_stem}_attention.jpg")
        cv2.imwrite(output_path, final_output)

        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            print(f"Processed {i + 1}/{len(image_files)}...")

    print(f"Heatmaps saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_folder()