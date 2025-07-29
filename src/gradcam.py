import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("models/resnet18_final.pth", map_location=device))
model.to(device)
model.eval()

# Select the target layer
target_layers = [model.layer4[-1]]

# Load image
img_path = "data\processed\split\test\intraoral_front\2018.10_2018.08.17_front.jpg"  # Replace with an actual test image path
rgb_img = np.array(Image.open(img_path).convert("RGB").resize((224, 224))) / 255.0
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# GradCAM
cam = GradCAM(model=model, target_layers=target_layers, device=device)
grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])
grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Save
os.makedirs("outputs/gradcam", exist_ok=True)
cv2.imwrite("outputs/gradcam/gradcam_result.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

