import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from torchvision import models, transforms

# Class order must match the order used during training: sorted class folder names.
CLASS_NAMES = [
    "SUV",
    "bus",
    "family sedan",
    "fire engine",
    "heavy truck",
    "jeep",
    "minibus",
    "racing car",
    "taxi",
    "truck",
]

MODEL_OPTIONS = {
    "ResNet50 (Frozen Layers) 93.5%": "best_model.pth",
    "ResNet50 (Unfrozen Layers) 97.5%": "best_model_unfreeze.pth",
}

HEAVY_VEHICLES = {"heavy truck", "bus", "minibus", "truck"}
EMERGENCY_VEHICLES = {"fire engine"}
NORMAL_VEHICLES = {"suv", "family sedan", "jeep", "racing car", "taxi"}

NOT_VEHICLE_THRESHOLD = 0.40


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model(model_path: str, device_name: str) -> torch.nn.Module:
    """Load a ResNet50 model checkpoint once per Streamlit session."""
    device = torch.device(device_name)
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_inference_transform() -> transforms.Compose:
    """Deterministic preprocessing for production inference."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def predict(image: Image.Image, model: torch.nn.Module, device: torch.device):
    transform = build_inference_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    return CLASS_NAMES[pred.item()], confidence.item()


def play_category_alert(predicted_label: str) -> None:
    label = predicted_label.lower()
    if label in HEAVY_VEHICLES:
        st.audio("beep.mp3")
    elif label in EMERGENCY_VEHICLES:
        st.audio("beep2.mp3")
    elif label in NORMAL_VEHICLES:
        st.audio("beep3.mp3")


st.set_page_config(page_title="Vehicle Classification", page_icon="🚗")
st.title("Vehicle Classification App")
st.write(f"This app can classify vehicles into {len(CLASS_NAMES)} classes: {', '.join(CLASS_NAMES)}.")
st.write("Upload an image of a vehicle to classify it.")

selected_model = st.selectbox("Select Model:", list(MODEL_OPTIONS.keys()))
model_path = MODEL_OPTIONS[selected_model]
device = get_device()
model = load_model(model_path, str(device))

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    predicted_class, confidence = predict(image, model, device)

    if confidence < NOT_VEHICLE_THRESHOLD:
        predicted_label = "not a vehicle"
        st.audio("beep4.mp3")
    else:
        predicted_label = predicted_class
        play_category_alert(predicted_label)

    st.write(f"Confidence: {confidence:.4f}")
    st.write(f"Predicted Class: This is {predicted_label}.")
