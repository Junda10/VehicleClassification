import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models

train_dir='C:/Dataset/vehicleClass/train/'
val_dir='C:/Dataset/vehicleClass/val/'
test_dir='C:/Dataset/vehicleClass/test'

classes=[]
paths=[]
for dirname, _, filenames in os.walk(train_dir):
    for filename in filenames:
        classes+=[dirname.split('/')[-1]]
        paths+=[(os.path.join(dirname, filename))]
        
tclasses=[]
tpaths=[]
for dirname, _, filenames in os.walk(val_dir):
    for filename in filenames:
        tclasses+=[dirname.split('/')[-1]]
        tpaths+=[(os.path.join(dirname, filename))]
        
#Creating Class Name Mappings
N=list(range(len(classes)))
class_names=sorted(set(classes))
normal_mapping=dict(zip(class_names,N)) 
reverse_mapping=dict(zip(N,class_names))       

#Creating DataFrames with Paths, Classes, and Labels
data=pd.DataFrame(columns=['path','class','label'])
data['path']=paths
data['class']=classes
data['label']=data['class'].map(normal_mapping)

tdata=pd.DataFrame(columns=['path','class','label'])
tdata['path']=tpaths
tdata['class']=tclasses
tdata['label']=tdata['class'].map(normal_mapping)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Load the model
best_model = models.resnet50(pretrained=False)
num_classes = 10
best_model.fc = torch.nn.Linear(best_model.fc.in_features, num_classes)
best_model.load_state_dict(torch.load('C:/Dataset/vehicleClass/models/best_model.pth', map_location=device))
best_model = best_model.to(device)
best_model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
        predicted_label = class_names[pred.item()]  # Get predicted class name

    # Display prediction
    st.write(f"Predicted Class: {predicted_label}")

# To run this app, save it as 'app.py' and use the command: streamlit run app.py
