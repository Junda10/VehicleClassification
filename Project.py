import os
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load the best_model (if saved)
best_model_path = 'best_model.pth'  # Re-instantiate your model
best_model = ...  # Re-instantiate your model class here
best_model.load_state_dict(torch.load(best_model_path))
best_model = best_model.to(device)
best_model.eval()

# Define the transform (same as used for training)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit application layout
st.title("Vehicle Classification")
st.write("Upload your vehicle image below:")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Process the uploaded image
if uploaded_file is not None:
    # Load and transform the image
    img = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

    # Make prediction
    with torch.no_grad():
        output = best_model(img_tensor)
        _, pred = torch.max(output, 1)
        predicted_label = class_names[pred.item()]  # Store the predicted class name

    # Display the image and prediction
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Class: {predicted_label}")

# Run the Streamlit app by executing: streamlit run your_script.py
