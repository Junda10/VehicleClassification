import streamlit as st
import os
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import time  # For generating unique filenames
import torch.nn.functional as F  # For softmax
import requests  # For uploading to GitHub
import base64  # For encoding image files

# ===== Directories for training data and predictions =====
train_dir = 'vehicleClass/train/'
predictions_dir = 'vehicleClass/predictions/'
os.makedirs(predictions_dir, exist_ok=True)  # Create the predictions directory if not exists

# ===== Class and Path Mapping =====
classes = []
paths = []
for dirname, _, filenames in os.walk(train_dir):
    for filename in filenames:
        classes.append(dirname.split('/')[-1])  # Get class name from folder name
        paths.append(os.path.join(dirname, filename))  # Get file path

class_names = sorted(set(classes))  # Get unique class names
normal_mapping = {name: index for index, name in enumerate(class_names)}

# Create DataFrame with file paths and labels
data = pd.DataFrame({'path': paths, 'class': classes})
data['label'] = data['class'].map(normal_mapping)

# ===== Check Device Availability =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Streamlit App Header =====
st.title("Vehicle Classification App")
st.write(f"Classify vehicles into these categories: {class_names}")
st.write("Upload an image of a vehicle to begin.")

# ===== Model Selection =====
model_options = {
    "ResNet50 (Frozen Layers) 93.5%": "best_model.pth",
    "ResNet50 (Unfrozen Layers) 98.0%": "best_model_unfreeze.pth"
}
selected_model = st.selectbox("Select Model:", list(model_options.keys()))

# ===== Load Selected Model =====
model_path = model_options[selected_model]
best_model = models.resnet50(pretrained=True)
best_model.fc = torch.nn.Linear(best_model.fc.in_features, len(class_names))  # Update final layer

best_model.load_state_dict(torch.load(model_path, map_location=device))
best_model = best_model.to(device)
best_model.eval()

# ===== Image Transformations =====
transform = transforms.Compose([
    transforms.RandomRotation(10),  # Rotate +/- 10 degrees
    transforms.RandomHorizontalFlip(),  # 50% chance to flip horizontally
    transforms.Resize(224),  # Resize shortest side to 224 pixels
    transforms.CenterCrop(224),  # Crop center to 224x224 pixels
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===== File Uploader for Image =====
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict the class
    with torch.no_grad():
        output = best_model(img_tensor)
        probabilities = F.softmax(output, dim=1)  # Get probabilities
        max_prob, pred = torch.max(probabilities, 1)  # Get max probability and prediction index

    # Thresholds for classification
    unknown_threshold = 0.4
    not_vehicle_threshold = 0.2

    # Determine predicted label based on confidence
    if max_prob.item() < not_vehicle_threshold:
        predicted_label = "This is not a vehicle."
    elif max_prob.item() < unknown_threshold:
        predicted_label = "Unknown vehicle."
    else:
        predicted_label = class_names[pred.item()]

    # Display predicted label
    st.write(f"Predicted Class: {predicted_label}")

    # Save the prediction image locally
    timestamp = int(time.time())  # Unique timestamp for filenames
    filename = f"{predicted_label.replace(' ', '_')}_{timestamp}.jpg"
    save_path = os.path.join(predictions_dir, filename)
    image.save(save_path)

# ===== Function to Upload to GitHub =====
def upload_to_github(file_path, repo, branch, token):
    """Upload the saved image to GitHub."""
    with open(file_path, 'rb') as f:
        content = base64.b64encode(f.read()).decode('utf-8')

    url = f"https://api.github.com/repos/{repo}/contents/{os.path.basename(file_path)}"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    data = {
        "message": "Upload prediction image",  # Commit message
        "content": content,
        "branch": branch
    }
    
    response = requests.put(url, headers=headers, json=data)

    if response.status_code == 201:
        st.write("File successfully uploaded to GitHub.")
    else:
        st.write(f"Failed to upload: {response.json()}")

    # Upload the image to GitHub
    upload_to_github(
        file_path=save_path,
        repo="Junda10/VehicleClassification",  # Replace with your repo
        branch="main",  # Replace with the correct branch
        token="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"  # Replace with your GitHub token
    )

