# Demo.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import io
import os
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from langchain_openai import ChatOpenAI
import torch.nn.functional as F

# --------------------------
# CONFIG + ENV
# --------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # for ChatOpenAI
HF_REPO_ID = "Ashutosh1975/plant-disease-model"       # your HF repo

if not OPENROUTER_API_KEY:
    st.warning("⚠️ OPENROUTER_API_KEY not found in environment variables. LLM responses will not work.")

# Initialize LLM (safe: only if key exists)
llm = None
if OPENROUTER_API_KEY:
    try:
        llm = ChatOpenAI(
            model_name="openai/gpt-4o-mini",
            temperature=0.7,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1"
        )
    except Exception as e:
        st.warning(f"LLM init failed: {e}")
        llm = None

# --------------------------
# MODEL ARCHITECTURE
# --------------------------
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((14, 14))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------
# LABELS (must match models)
# --------------------------
dataset_classes = {
    "Main Model": [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ],
    "Apple Model": ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'],
    "Berry Model": ['Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Raspberry___healthy', 'Strawberry___Leaf_scorch', 'Strawberry___healthy'],
    "Corn Model": ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy'],
    "Grapes Model": ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy'],
    "Peach Model": ['Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy'],
    "Potato Model": ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
    "Soybean Model": ['Soybean___healthy', 'Squash___Powdery_mildew'],
    "Tomato Model": ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                     'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
}

# --------------------------
# Mapping HF filenames (exact names in your repo)
# --------------------------
MODEL_FILES = {
    "Main Model": "plant_disease_cnn.pth",
    "Apple Model": "plant_disease_Apple.pth",
    "Berry Model": "plant_disease_Berry.pth",
    "Corn Model": "plant_disease_Corn.pth",
    "Grapes Model": "plant_disease_Graps.pth",
    "Peach Model": "plant_disease_Peach.pth",
    "Potato Model": "plant_disease_Potato.pth",
    "Soybean Model": "plant_disease_Soybean.pth",
    "Tomato Model": "plant_disease_Tomato.pth"
}

# --------------------------
# Image transform
# --------------------------
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)


# --------------------------
# SAFE HUB LOADER (cached by Streamlit)
# --------------------------
@st.cache_resource(show_spinner=False)
def load_model_from_hub(model_filename: str, num_classes: int):
    """
    Downloads (if needed) the file from HuggingFace hub and loads
    it into the PlantDiseaseCNN architecture. Cached across app lifetime.
    """
    try:
        st.info(f"Downloading model: {model_filename} ")
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=model_filename, force_download=False)
    except Exception as e:
        st.error(f"Failed to download `{model_filename}` from HuggingFace Hub.")
        st.text(str(e))
        return None

    # build the model architecture for required num_classes
    model = PlantDiseaseCNN(num_classes)

    try:
        checkpoint = torch.load(model_path, map_location="cpu")

        # If checkpoint is a dict (state_dict), safely load matching keys
        if isinstance(checkpoint, dict):
            model_state = model.state_dict()
            filtered = {k: v for k, v in checkpoint.items() if k in model_state and v.shape == model_state[k].shape}
            model_state.update(filtered)
            model.load_state_dict(model_state)
        else:
            # Check for PyTorch serialized model object with .state_dict()
            try:
                state = checkpoint.state_dict()
                model.load_state_dict(state, strict=False)
            except Exception:
                # as a last resort try to use it directly (may override)
                model = checkpoint
    except Exception as e:
        st.error("⚠️ Error loading checkpoint into model architecture.")
        st.text(str(e))
        return None

    model.eval()
    return model


# --------------------------
# PREDICTION
# --------------------------
def get_prediction(model: nn.Module, image_bytes: bytes, classes: list):
    tensor = transform_image(image_bytes)
    with torch.no_grad():
        outputs = model(tensor)
        # If model returns logits: compute softmax
        try:
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        except Exception:
            # try converting outputs to tensor then softmax
            probs = F.softmax(torch.tensor(outputs), dim=1).cpu().numpy()[0]
        top_idx = int(probs.argmax())
        return classes[top_idx], float(probs[top_idx])


# --------------------------
# LLM / Prompt helper
# --------------------------
PLANT_PROMPT = """
You are an expert plant pathologist and agronomist.
Given the disease name below, provide a short clear explanation structured into
1) Overview
2) Symptoms
3) Spread
4) Recommended immediate actions
5) Prevention tips
Keep it concise and actionable for a farmer/user.
Disease: {disease}
"""

def ask_llm(disease_name: str):
    if llm is None:
        return "💡 LLM not configured (OPENROUTER_API_KEY missing). Only model prediction is shown."
    prompt = PLANT_PROMPT.format(disease=disease_name)
    try:
        resp = llm.invoke(prompt)
        # langchain_openai ChatOpenAI returns obj with .content (from earlier usage); handle gracefully
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"LLM error: {e}"


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Plant Disease Detection (Online models)", layout="wide")
st.title("🌱 Plant Disease Detection ")

st.sidebar.header("Model & Options")
model_choice = st.sidebar.selectbox("Choose model:", list(MODEL_FILES.keys()))
show_image = st.sidebar.checkbox("Show uploaded image", value=True)
save_history = st.sidebar.checkbox("Save prediction to local history (results.csv)", value=True)

# prepare mapping
classes = dataset_classes[model_choice]
num_classes = len(classes)
model_file = MODEL_FILES[model_choice]

# load model (cached)
with st.spinner(f"Loading model `{model_file}` from HuggingFace..."):
    model = load_model_from_hub(model_file, num_classes)

if model is None:
    st.error("Model could not be loaded. Check HF repo name and filenames.")
    st.stop()

uploaded_file = st.file_uploader("📸 Upload a plant leaf image", type=["png","jpg","jpeg","bmp","tiff"])

if uploaded_file:
    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_resized = image.resize((256,256))
        if show_image:
            st.image(image_resized, caption="Uploaded (resized) image", use_container_width=False)
    except UnidentifiedImageError:
        st.error("Uploaded file is not a valid image.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading image: {e}")
        st.stop()

    if st.button("🔍 Predict"):
        try:
            pred_label, confidence = get_prediction(model, image_bytes, classes) # type: ignore
            st.success(f"🧩 Prediction: **{pred_label}**  —  Confidence: **{confidence:.3f}**")

            # Ask LLM for human-friendly explanation
            with st.spinner("Asking LLM for explanation..."):
                llm_result = ask_llm(pred_label)
            st.subheader("🧠 AI Explanation (LLM)")
            st.markdown(llm_result)

            # Save to results.csv if enabled
            if save_history:
                results_path = "results.csv"
                row = {
                    "model": model_choice,
                    "prediction": pred_label,
                    "confidence": float(confidence),
                    "file_name": uploaded_file.name,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
                df_row = pd.DataFrame([row])
                if os.path.exists(results_path):
                    df_row.to_csv(results_path, mode="a", header=False, index=False)
                else:
                    df_row.to_csv(results_path, index=False)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# --------------------------
# Prediction history viewer
# --------------------------
st.markdown("---")
st.header("📊 Prediction History")
if os.path.exists("results.csv"):
    try:
        df_hist = pd.read_csv("results.csv")
        st.dataframe(df_hist.tail(20))
    except Exception as e:
        st.error(f"Could not read results.csv: {e}")
else:
    st.info("No prediction history found yet.")
