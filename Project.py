import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import pandas as pd
import time  # To manage alerts
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

# DataFrame with Paths, Classes, and Labels
data = pd.DataFrame({'path': paths, 'class': classes})
data['label'] = data['class'].map(normal_mapping)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Streamlit app
st.title("Vehicle Classification App")
st.write(f"This App can classify vehicles into these classes: {class_names}")
st.write("Upload an image of a vehicle to classify it.")

# Model selection dropdown
model_options = {
    "ResNet50 (Frozen Layers) 93.5%": "best_model.pth",
    "ResNet50 (Unfrozen Layers) 98.0%": "best_model_unfreeze.pth"
}
selected_model = st.selectbox("Select Model:", list(model_options.keys()))

# Load the selected model
model_path = model_options[selected_model]
best_model = models.resnet50(pretrained=True)
num_classes = len(class_names)  # Use the actual number of classes
best_model.fc = torch.nn.Linear(best_model.fc.in_features, num_classes)

# Load model weights
best_model.load_state_dict(torch.load(model_path, map_location=device))
best_model = best_model.to(device)
best_model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Alert management
alert_interval = 300  # 5 minutes (in seconds)
last_alert_time = 0  # Store the timestamp of the last alert

heavy_vehicles = ['heavy truck', 'bus', 'minibus', 'truck']
emergency_vehicles = ['fire engine']
normal_vehicles = ['SUV', 'family sedan', 'jeep', 'racing car', 'taxi']

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Make a prediction
    with torch.no_grad():
        output = best_model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        max_prob, pred = torch.max(probabilities, 1)

    # Confidence thresholds
    unknown_threshold = 0.3
    not_vehicle_threshold = 0.2

    # Determine the prediction label
    if max_prob.item() < not_vehicle_threshold:
        predicted_label = "not a vehicle."
    elif max_prob.item() < unknown_threshold:
        predicted_label = "unknown vehicle."
    else:
        predicted_label = class_names[pred.item()]  # Get the class name

        # Check if the detected vehicle belongs to alert classes
        current_time = time.time()
        
        if predicted_label.lower() in heavy_vehicles:
            st.audio("beep.mp3")  # Provide the path to your sound file

        # Check if the predicted class is a fire engine
        if predicted_label.lower() in emergency_vehicles:
            st.audio("beep2.mp3")  # Provide the path to your fire engine sound file
            
        if predicted_label.lower() in normal_vehicles:
            st.audio("beep3.mp3")  # Provide the path to your fire engine sound file

    # Display prediction and threshold
    st.write(f"Threshold: {max_prob.item()}")
    st.write(f"Predicted Class: This is {predicted_label}")
