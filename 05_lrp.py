import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# FIX 1: Set local cache directory to avoid Permission Errors
os.environ['TORCH_HOME'] = './torch_cache'

# FIX 2: Check for CUDA (GPU) and fallback to CPU if not available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Using weights='DEFAULT' instead of pretrained=True to avoid warnings
        self.vgg16 = models.vgg16(weights='DEFAULT')
        self.features = self.vgg16.features
        self.avgpool = self.vgg16.avgpool
        self.classifier = self.vgg16.classifier
        
        # Placeholder for gradients in Grad-CAM
        self.gradients = None
    
    # Hook for Grad-CAM to extract gradients from the last convolutional layer
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Extract features and register hook on the last conv layer for Grad-CAM
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 30: # Last conv layer in VGG16 features
                x.register_hook(self.activations_hook)
                self.last_conv_layer_output = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize model and move to appropriate device (CPU or GPU)
model = CNNModel()
model.to(device)
model.eval()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

def simple_lrp(model, input_image):
    """
    Implementation of Layer-wise Relevance Propagation (Simple Rule)
    """
    input_image.requires_grad = True
    output = model(input_image)
    
    target_class = output.argmax().item()
    model.zero_grad()
    output[0, target_class].backward(retain_graph=True)
    
    relevance = input_image.grad.data[0].cpu().numpy()
    relevance = np.maximum(0, relevance) 
    relevance = np.sum(relevance, axis=0) 
    
    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-8)
    return relevance

def grad_cam(model, input_image):
    """
    Implementation of Grad-CAM (Gradient-weighted Class Activation Mapping)
    """
    output = model(input_image)
    target_class = output.argmax().item()
    
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get pooled gradients
    gradients = model.gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # Get activations from the last conv layer
    activations = model.last_conv_layer_output.detach()
    
    # Weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    # Average the channels and apply ReLU
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    
    # Normalize the heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

# Example usage: Ensure you have organized your data folders as per Task 2
# Update the 'image_path' below to a real image file in your folder.
image_path = "data/brain_mri/testing/glioma_tumor/image.jpg" 

if os.path.exists(image_path):
    print(f"Processing image: {image_path}")
    img_tensor = preprocess_image(image_path)
    
    # Generate heatmaps
    lrp_map = simple_lrp(model, img_tensor)
    gradcam_map = grad_cam(model, img_tensor)
    
    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display Original
    original_img = Image.open(image_path).resize((224, 224))
    ax[0].imshow(original_img)
    ax[0].set_title("Original MRI")
    ax[0].axis('off')
    
    # Display LRP
    ax[1].imshow(lrp_map, cmap='hot')
    ax[1].set_title("LRP Heatmap")
    ax[1].axis('off')
    
    # Display Grad-CAM
    ax[2].imshow(gradcam_map, cmap='jet')
    ax[2].set_title("Grad-CAM Heatmap")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Visualizations generated successfully.")
else:
    print(f"Image not found at: {image_path}")
    print("Please ensure you have completed Task 2 (Folder Organization) and the path is correct.")

print("LRP & Grad-CAM methods loaded successfully on " + str(device))