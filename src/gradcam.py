from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch
import cv2
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("models/resnet18_final.pth", map_location=device))
model.to(device)
model.eval()

# Target layer
target_layers = [model.layer4[-1]]

# Load and preprocess image
img_path = r"C:\Users\SSD\orthodontic_image_classification\data\processed\split\test\intraoral_front\2018.10_2018.08.17_front.jpg"
image = Image.open(img_path).convert("RGB").resize((224, 224))
rgb_img = np.array(image).astype(np.float32) / 255.0

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(image).unsqueeze(0).to(device)

# Predict class
with torch.no_grad():
    outputs = model(input_tensor)
    predicted_class = torch.argmax(outputs, dim=1).item()

# Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_class)])[0]

# Overlay CAM on original image
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Save result
os.makedirs("outputs/gradcam", exist_ok=True)
cv2.imwrite("outputs/gradcam/gradcam_result.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
