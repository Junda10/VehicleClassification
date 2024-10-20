import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import pandas as pd

# Directories for training data
train_dir = 'vehicleClass/train/'

# Create class and path mappings
classes = []
paths = []
for dirname, _, filenames in os.walk(train_dir):
    for filename in filenames:
        classes.append(dirname.split('/')[-1])  # Get class name from directory name
        paths.append(os.path.join(dirname, filename))  # Get file path

# Create Class Name Mappings
class_names = sorted(set(classes))
st.write(f"Class Names: {class_names}")  # Print class names for debugging
normal_mapping = {name: index for index, name in enumerate(class_names)} 

# Creating DataFrame with Paths, Classes, and Labels
data = pd.DataFrame({'path': paths, 'class': classes})
data['label'] = data['class'].map(normal_mapping)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Model selection dropdown
model_options = {
    "ResNet50 (Frozen Layers)": "best_model.pth",
    "ResNet50 (Unfrozen Layers)": "best_model_unfreeze.pth"
}
selected_model = st.selectbox("Select Model:", list(model_options.keys()))

# Load the model based on user selection
model_path = model_options[selected_model]
best_model = models.resnet50(pretrained=True)
num_classes = len(class_names)  # Use the actual number of classes
best_model.fc = torch.nn.Linear(best_model.fc.in_features, num_classes)

# Load the corresponding model weights
best_model.load_state_dict(torch.load(model_path, map_location=device))
best_model = best_model.to(device)
best_model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.RandomRotation(10),      # Rotate +/- 10 degrees
    transforms.RandomHorizontalFlip(),  # Flip 50% of images
    transforms.Resize(224),             # Resize shortest side to 224 pixels
    transforms.CenterCrop(224),         # Crop longest side to 224 pixels at center
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Streamlit application
st.title("Vehicle Classification App")
st.write("Upload an image of a vehicle to classify it.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = best_model(img_tensor)
        _, pred = torch.max(output, 1)
        predicted_label = class_names[pred.item()]  # Get the predicted class name

    st.write(f"Predicted Class: {predicted_label}")  # Display predicted label
