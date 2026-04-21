import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import io
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Get API key from env
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("❌ OPENROUTER_API_KEY not found in environment variables. Please set it correctly.")

# Initialize the LLM once globally with API key and base URL
llm = ChatOpenAI(
    model_name="openai/gpt-4o-mini",
    temperature=0.7,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1"
)


# ----------------------------
# Model Definition
# ----------------------------
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((14, 14))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*14*14, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------------
# Dataset-Specific Classes
# ----------------------------
dataset_classes = {
    "Main Model": [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ],
    "Apple Model": ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'],
    "Berry Model": ['Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Raspberry___healthy','Strawberry___Leaf_scorch','Strawberry___healthy'],
    "Corn Model": ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy'],
    "Grapes Model": ['Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy'],
    "Peach Model": ['Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy'],
    "Potato Model": ['Potato___Early_blight','Potato___Late_blight','Potato___healthy'],
    "Soybean Model": ['Soybean___healthy','Squash___Powdery_mildew'],
    "Tomato Model": ['Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']
}

# ----------------------------
# Image Transform
# ----------------------------
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# ----------------------------
# Model Load & Prediction
# ----------------------------
def load_model(model_path, num_classes):
    model = PlantDiseaseCNN(num_classes)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model

def get_prediction(model, image_bytes, classes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, predicted = torch.max(outputs.data, 1)
    return classes[predicted.item()]

# -------------------------------------------------------------
def get_openai_response(prompt):
    response = llm.invoke(prompt)
    return response

plant_disease_prompt_template = """
You are a highly knowledgeable plant pathologist and agricultural expert.

Based only on the following predicted disease name:

{predicted_disease}

Provide a detailed, clear, and structured explanation covering:

1. A brief overview of the disease.
2. Detailed symptoms commonly associated with it.
3. How the disease typically spreads.
4. Recommended immediate actions and preventive measures.
5. Additional useful tips for plant growers.

Please format your response with clear headings as shown below:

---
Plant Disease Report: {predicted_disease}

1. Overview:
[Brief description of the disease.]

2. Symptoms to Look For:
- Symptom 1
- Symptom 2
- Symptom 3

3. How the Disease Spreads:
[Explanation.]

4. Recommended Actions:
- Action 1
- Action 2
- Action 3

5. Additional Tips:
[Any helpful info.]

---

Make sure your answer is accurate, easy to understand, and helpful for a plant grower trying to manage this disease.
"""



# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Plant Disease + AI Chat", layout="wide")
st.title("🌱 Plant Disease Detection + AI Assistant")

st.write("Upload a plant leaf image and detect its disease using different trained models.")

# Sidebar
st.sidebar.header("⚙️ Settings")
st.sidebar.info("Choose a model below. Default is the Main Model trained on all datasets.")
model_choice = st.sidebar.selectbox("Choose a model:", list(dataset_classes.keys()))

BASE_MODEL_PATH = "/home/ashutosh/Desktop/Agri-Care-fullStack/ml_models/pytorch_models"
model_map = {
    "Main Model": os.path.join(BASE_MODEL_PATH, "plant_disease_cnn.pth"),
    "Apple Model": os.path.join(BASE_MODEL_PATH, "plant_disease_Apple.pth"),
    "Berry Model": os.path.join(BASE_MODEL_PATH, "plant_disease_Berry.pth"),
    "Corn Model": os.path.join(BASE_MODEL_PATH, "plant_disease_Corn.pth"),
    "Grapes Model": os.path.join(BASE_MODEL_PATH, "plant_disease_Graps.pth"),
    "Peach Model": os.path.join(BASE_MODEL_PATH, "plant_disease_Peach.pth"),
    "Potato Model": os.path.join(BASE_MODEL_PATH, "plant_disease_Potato.pth"),
    "Soybean Model": os.path.join(BASE_MODEL_PATH, "plant_disease_Soybean.pth"),
    "Tomato Model": os.path.join(BASE_MODEL_PATH, "plant_disease_Tomato.pth")
}

model_path = model_map[model_choice]
if not os.path.exists(model_path):
    st.error(f"❌ Model file `{model_path}` not found!")
    st.stop()

classes = dataset_classes[model_choice]
num_classes = len(classes)
model = load_model(model_path, num_classes)

# File uploader
uploaded_file = st.file_uploader("📸 Upload an image...")  # any type allowed

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        
        # Open and resize the image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((256, 256), Image.ANTIALIAS)
        
        # Convert resized image back to bytes for prediction
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        resized_bytes = buf.getvalue()
        
        show_image_checkbox = st.checkbox("🖼 Show Image before Predicting")
        
        if show_image_checkbox:
            st.image(image, caption="Uploaded & Resized Image", use_container_width=True)
        
        if st.button("🔍 Predict"):
            prediction = None  # Initialize to avoid undefined variable
            response = None

            try:
                # Step 1: Try to get prediction from your CNN model
                try:
                    prediction = get_prediction(model, resized_bytes, classes)
                    st.info(f"🧩 Model Prediction: {prediction}")
                except Exception as e:
                    st.error("❌ Prediction model not working properly. Please check model weights or input format.")
                    st.stop()

                # Step 2: Prepare chatbot prompt
                full_prompt = plant_disease_prompt_template.format(predicted_disease=prediction)

                # Step 3: Try chatbot (OpenRouter or OpenAI)
                try:
                    response = get_openai_response(full_prompt)
                except Exception:
                    st.warning("⚠️ Chatbot model not responding — possible issues: invalid API key, insufficient credits, or server downtime.")
                    response = None

                # Step 4: Show analysis
                st.subheader("🧠 Analysis Result")
                if response is None:
                    st.info("💡 Displaying only the CNN prediction result:")
                    st.success(f"Predicted Disease: {prediction}")
                else:
                    try:
                        # Try to safely show response
                        st.success("✅ AI Response:")
                        st.markdown(response.content if hasattr(response, "content") else str(response))
                    except Exception:
                        st.warning("⚠️ Could not parse chatbot response. Showing only prediction.")
                        st.success(f"Predicted Disease: {prediction}")

            except torch.cuda.CudaError:
                st.error("⚠️ GPU processing error — try running on CPU mode.")
            except UnidentifiedImageError:
                st.error("❌ Invalid image file. Please upload a valid image.")
            except Exception as e:
                st.error("❌ Unexpected error during prediction or chatbot response.")
                st.text(f"Details: {e}")
    except Exception as e:
        st.error("❌ Error reading uploaded image. Please try again.")
        
# ==============================
# 📊 Prediction History Section
# ==============================

if st.button("📊 Show Prediction History"):
    RESULTS_FILE = "results.csv"
    try:
        if os.path.exists(RESULTS_FILE):
            df = pd.read_csv(RESULTS_FILE)

            if df.empty:
                st.warning("📂 The results file exists but is empty.")
            else:
                st.success("✅ Showing the last 10 predictions:")
                st.dataframe(df.tail(10))
        else:
            st.info("ℹ️ No prediction history available. Make a prediction first!")

    except pd.errors.EmptyDataError:
        st.error("⚠️ The results file is empty or corrupted. Please try again after a new prediction.")
    except pd.errors.ParserError:
        st.error("❌ Error reading results.csv — file format invalid or corrupted.")
    except Exception as e:
        st.error("❌ Unexpected error while loading prediction history.")
        st.text(f"Details: {e}")


