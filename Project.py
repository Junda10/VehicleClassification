import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import pandas as pd
import time  # To generate unique filenames
import torch.nn.functional as F  # For softmax

# Directories for training data and storing predictions
train_dir = 'vehicleClass/train/'
predictions_dir = 'vehicleClass/predictions/'

# Create the predictions directory if it doesn't exist
os.makedirs(predictions_dir, exist_ok=True)

# Create class and path mappings
classes = []
paths = []
for dirname, _, filenames in os.walk(train_dir):
    for filename in filenames:
        classes.append(dirname.split('/')[-1])  # Get class name from directory name
        paths.append(os.path.join(dirname, filename))  # Get file path

# Create Class Name Mappings
class_names = sorted(set(classes))
normal_mapping = {name: index for index, name in enumerate(class_names)}

# Creating DataFrame with Paths, Classes, and Labels
data = pd.DataFrame({'path': paths, 'class': classes})
data['label'] = data['class'].map(normal_mapping)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit application
st.title("Vehicle Classification App")
st.write(f"This App is able to classify the vehicle into these classes: {class_names}")  # Print class names for debugging
st.write("Upload an image of a vehicle to classify it.")

# Model selection dropdown
model_options = {
    "ResNet50 (Frozen Layers) 93.5%": "best_model.pth",
    "ResNet50 (Unfrozen Layers) 98.0%": "best_model_unfreeze.pth"
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
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        max_prob, pred = torch.max(probabilities, 1)  # Get the highest probability and corresponding index

    # Set thresholds for recognizing "Unknown" and "Not a Vehicle"
    unknown_threshold = 0.6  # Below this, we mark as unknown
    not_vehicle_threshold = 0.3  # Below this, we assume it's not a vehicle

    # Determine the predicted label based on confidence levels
    if max_prob.item() < not_vehicle_threshold:
        predicted_label = "This is not a vehicle."
    elif max_prob.item() < unknown_threshold:
        predicted_label = "Unknown vehicle."
    else:
        predicted_label = class_names[pred.item()]  # Get the predicted class name

    st.write(f"Predicted Class: {predicted_label}")  # Display predicted label

    # Save the image with prediction
    timestamp = int(time.time())  # Use timestamp to ensure unique filenames
    filename = f"{predicted_label.replace(' ', '_')}_{timestamp}.jpg"
    save_path = os.path.join(predictions_dir, filename)
    image.save(save_path)
    st.write(f"Saving image to: {save_path}")

