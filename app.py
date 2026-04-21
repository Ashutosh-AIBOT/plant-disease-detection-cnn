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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_REPO_ID = "Ashutosh1975/plant-disease-model"

# --------------------------
# PROFESSIONAL UI STYLING
# --------------------------
st.set_page_config(
    page_title="PhytoScan — Plant Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg-primary: #0a0f0a;
    --bg-secondary: #111811;
    --bg-card: #141f14;
    --bg-card-hover: #1a2a1a;
    --accent-green: #4ade80;
    --accent-lime: #a3e635;
    --accent-emerald: #34d399;
    --text-primary: #f0fdf4;
    --text-secondary: #86efac;
    --text-muted: #4ade8088;
    --border: #1f3a1f;
    --border-accent: #4ade8033;
    --danger: #f87171;
    --warning: #fbbf24;
    --shadow: 0 0 40px #4ade8011;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse at 20% 20%, #4ade8008 0%, transparent 60%),
        radial-gradient(ellipse at 80% 80%, #34d39906 0%, transparent 60%);
}

/* ---- SIDEBAR ---- */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stCheckbox label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 10px !important;
}

/* ---- HEADER ---- */
.phyto-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 32px 0 24px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
.phyto-logo {
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, #4ade80, #34d399);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 24px #4ade8044;
}
.phyto-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
    line-height: 1;
    letter-spacing: -0.02em;
}
.phyto-subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 4px 0 0 0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 400;
}

/* ---- CARDS ---- */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow);
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--accent-green);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---- UPLOAD ZONE ---- */
.upload-zone {
    background: var(--bg-card);
    border: 2px dashed var(--border-accent);
    border-radius: 20px;
    padding: 48px 24px;
    text-align: center;
    transition: all 0.3s ease;
    margin-bottom: 20px;
}
.upload-zone:hover {
    border-color: var(--accent-green);
    background: var(--bg-card-hover);
}
.upload-icon {
    font-size: 3rem;
    margin-bottom: 12px;
    display: block;
}
.upload-text {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 500;
    margin: 0;
}
.upload-hint {
    color: var(--text-muted);
    font-size: 0.8rem;
    margin-top: 6px;
}

/* ---- PREDICTION RESULT ---- */
.result-box {
    background: linear-gradient(135deg, #0f2a1a, #0a1f12);
    border: 1px solid var(--accent-green);
    border-radius: 16px;
    padding: 28px;
    margin: 20px 0;
    box-shadow: 0 0 30px #4ade8022, inset 0 1px 0 #4ade8022;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent-green);
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}
.result-confidence {
    font-size: 0.85rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}
.confidence-bar-bg {
    flex: 1;
    height: 6px;
    background: #1f3a1f;
    border-radius: 3px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #4ade80, #a3e635);
    border-radius: 3px;
    transition: width 1s ease;
}
.result-healthy {
    border-color: var(--accent-emerald);
    background: linear-gradient(135deg, #0a2a1f, #051a12);
    box-shadow: 0 0 30px #34d39922;
}
.result-healthy .result-label {
    color: var(--accent-emerald);
}
.result-disease {
    border-color: var(--warning);
    background: linear-gradient(135deg, #2a1f0a, #1a1205);
    box-shadow: 0 0 30px #fbbf2422;
}
.result-disease .result-label {
    color: var(--warning);
}

/* ---- TAGS ---- */
.tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.tag-healthy {
    background: #34d39922;
    color: var(--accent-emerald);
    border: 1px solid #34d39944;
}
.tag-disease {
    background: #fbbf2422;
    color: var(--warning);
    border: 1px solid #fbbf2444;
}
.tag-model {
    background: #4ade8011;
    color: var(--accent-green);
    border: 1px solid var(--border-accent);
}

/* ---- LLM SECTION ---- */
.llm-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-top: 20px;
}
.llm-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--accent-lime);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ---- STATS ROW ---- */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--accent-green);
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ---- HISTORY TABLE ---- */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}
.stDataFrame thead tr th {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
.stDataFrame tbody tr:hover {
    background: var(--bg-card-hover) !important;
}

/* ---- BUTTONS ---- */
.stButton > button {
    background: linear-gradient(135deg, #4ade80, #34d399) !important;
    color: #0a0f0a !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    padding: 12px 32px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px #4ade8033 !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px #4ade8055 !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ---- FILE UPLOADER ---- */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-accent) !important;
    border-radius: 16px !important;
    padding: 16px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-green) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ---- SPINNER ---- */
.stSpinner > div {
    border-top-color: var(--accent-green) !important;
}

/* ---- ALERTS ---- */
.stSuccess {
    background: #0f2a1a !important;
    border: 1px solid var(--accent-green) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}
.stError {
    background: #2a0f0f !important;
    border: 1px solid var(--danger) !important;
    border-radius: 12px !important;
}
.stWarning {
    background: #2a1f0a !important;
    border: 1px solid var(--warning) !important;
    border-radius: 12px !important;
}
.stInfo {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 12px !important;
}

/* ---- DIVIDER ---- */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 32px 0 !important;
}

/* ---- MARKDOWN ---- */
.stMarkdown p {
    color: var(--text-secondary);
    line-height: 1.7;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Syne', sans-serif;
    color: var(--text-primary);
}

/* ---- SIDEBAR MODEL BADGE ---- */
.model-badge {
    background: linear-gradient(135deg, #4ade8022, #34d39911);
    border: 1px solid var(--border-accent);
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 16px;
    text-align: center;
}
.model-badge-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: var(--accent-green);
    font-size: 0.95rem;
}
.model-badge-classes {
    color: var(--text-muted);
    font-size: 0.75rem;
    margin-top: 4px;
}

/* ---- IMAGE PREVIEW ---- */
.img-preview-wrapper {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border-accent);
    box-shadow: var(--shadow);
}

/* hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --------------------------
# LLM INIT
# --------------------------
if not OPENROUTER_API_KEY:
    st.sidebar.warning("⚠️ OPENROUTER_API_KEY missing — AI explanations disabled.")

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
        st.sidebar.warning(f"LLM init failed: {e}")

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
# LABELS
# --------------------------
dataset_classes = {
    "Main Model": [
        'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
        'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
        'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy',
        'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
        'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus','Tomato___healthy'
    ],
    "Apple Model": ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy'],
    "Berry Model": ['Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
                    'Raspberry___healthy','Strawberry___Leaf_scorch','Strawberry___healthy'],
    "Corn Model": ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy'],
    "Grapes Model": ['Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy'],
    "Peach Model": ['Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
                    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy'],
    "Potato Model": ['Potato___Early_blight','Potato___Late_blight','Potato___healthy'],
    "Soybean Model": ['Soybean___healthy','Squash___Powdery_mildew'],
    "Tomato Model": ['Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
                     'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
                     'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']
}

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

MODEL_ICONS = {
    "Main Model": "🌿", "Apple Model": "🍎", "Berry Model": "🍓",
    "Corn Model": "🌽", "Grapes Model": "🍇", "Peach Model": "🍑",
    "Potato Model": "🥔", "Soybean Model": "🌱", "Tomato Model": "🍅"
}

# --------------------------
# HELPERS
# --------------------------
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)


@st.cache_resource(show_spinner=False)
def load_model_from_hub(model_filename: str, num_classes: int):
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=model_filename, force_download=False)
    except Exception as e:
        st.error(f"Failed to download `{model_filename}` from HuggingFace Hub.")
        st.text(str(e))
        return None

    model = PlantDiseaseCNN(num_classes)
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            model_state = model.state_dict()
            filtered = {k: v for k, v in checkpoint.items()
                        if k in model_state and v.shape == model_state[k].shape}
            model_state.update(filtered)
            model.load_state_dict(model_state)
        else:
            try:
                state = checkpoint.state_dict()
                model.load_state_dict(state, strict=False)
            except Exception:
                model = checkpoint
    except Exception as e:
        st.error("⚠️ Error loading checkpoint.")
        st.text(str(e))
        return None

    model.eval()
    return model


def get_prediction(model, image_bytes, classes):
    tensor = transform_image(image_bytes)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        top_idx = int(probs.argmax())
        return classes[top_idx], float(probs[top_idx]), probs


PLANT_PROMPT = """
You are an expert plant pathologist and agronomist.
Given the disease name below, provide a concise structured explanation:
1) Overview
2) Symptoms
3) Spread
4) Recommended immediate actions
5) Prevention tips
Keep it practical and actionable for a farmer or agronomist.
Disease: {disease}
"""

@st.cache_data(show_spinner=False)
def ask_llm_cached(disease_name: str):
    """Cache LLM response per disease — avoids re-calling on re-render."""
    if llm is None:
        return "💡 LLM not configured (OPENROUTER_API_KEY missing)."
    prompt = PLANT_PROMPT.format(disease=disease_name)
    try:
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"LLM error: {e}"


def is_healthy(label: str) -> bool:
    return "healthy" in label.lower()


def format_label(label: str) -> str:
    return label.replace("___", " — ").replace("_", " ")


def get_history_stats():
    results_path = "results.csv"
    if not os.path.exists(results_path):
        return 0, 0, 0
    try:
        df = pd.read_csv(results_path)
        total = len(df)
        healthy = df["prediction"].str.contains("healthy", case=False).sum()
        diseased = total - healthy
        return total, int(healthy), int(diseased)
    except:
        return 0, 0, 0


# --------------------------
# SIDEBAR
# --------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:800;
                    color:#4ade80; letter-spacing:-0.01em;">🌿 PhytoScan</div>
        <div style="font-size:0.72rem; color:#4ade8066; text-transform:uppercase;
                    letter-spacing:0.12em; margin-top:2px;">Plant Disease AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.72rem; color:#86efac; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Select Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("", list(MODEL_FILES.keys()), label_visibility="collapsed")

    classes = dataset_classes[model_choice]
    icon = MODEL_ICONS.get(model_choice, "🌿")

    st.markdown(f"""
    <div class="model-badge">
        <div style="font-size:1.8rem; margin-bottom:6px;">{icon}</div>
        <div class="model-badge-name">{model_choice}</div>
        <div class="model-badge-classes">{len(classes)} disease classes</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem; color:#86efac; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Options</div>', unsafe_allow_html=True)
    show_image = st.checkbox("Show uploaded image", value=True)
    show_top3 = st.checkbox("Show top 3 predictions", value=True)
    save_history = st.checkbox("Save to history (CSV)", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem; color:#86efac; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;">Supported Classes</div>', unsafe_allow_html=True)
    for c in classes:
        healthy_dot = "🟢" if is_healthy(c) else "🟡"
        st.markdown(f'<div style="font-size:0.72rem; color:#4ade8099; padding:3px 0; border-bottom:1px solid #1f3a1f22;">{healthy_dot} {format_label(c)}</div>', unsafe_allow_html=True)


# --------------------------
# HEADER
# --------------------------
st.markdown("""
<div class="phyto-header">
    <div class="phyto-logo">🌿</div>
    <div>
        <div class="phyto-title">PhytoScan</div>
        <div class="phyto-subtitle">AI-Powered Plant Disease Detection System</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------
# STATS ROW
# --------------------------
total_scans, healthy_count, diseased_count = get_history_stats()
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{total_scans}</div><div class="stat-label">Total Scans</div></div>', unsafe_allow_html=True)
with col_s2:
    st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#34d399;">{healthy_count}</div><div class="stat-label">Healthy</div></div>', unsafe_allow_html=True)
with col_s3:
    st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#fbbf24;">{diseased_count}</div><div class="stat-label">Diseased</div></div>', unsafe_allow_html=True)
with col_s4:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{len(classes)}</div><div class="stat-label">Classes Active</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------
# LOAD MODEL
# --------------------------
num_classes = len(classes)
model_file = MODEL_FILES[model_choice]

with st.spinner(f"Loading {model_choice} from HuggingFace Hub..."):
    model = load_model_from_hub(model_file, num_classes)

if model is None:
    st.error("❌ Model could not be loaded. Please check your HuggingFace repo and filenames.")
    st.stop()

# --------------------------
# MAIN LAYOUT
# --------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card-title">📸 Upload Leaf Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop or browse a plant leaf image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        try:
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_resized = image.resize((320, 320))

            if show_image:
                st.markdown('<div class="img-preview-wrapper">', unsafe_allow_html=True)
                st.image(image_resized, use_container_width=True, caption="")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div style="display:flex; gap:8px; margin-top:12px; flex-wrap:wrap;">
                <span class="tag tag-model">{icon} {model_choice}</span>
                <span class="tag tag-model">📄 {uploaded_file.name}</span>
                <span class="tag tag-model">📐 {image.size[0]}×{image.size[1]}px</span>
            </div>
            """, unsafe_allow_html=True)

        except UnidentifiedImageError:
            st.error("❌ Uploaded file is not a valid image.")
            st.stop()
        except Exception as e:
            st.error(f"Error reading image: {e}")
            st.stop()

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- PREDICT BUTTON ----
        # Use session state to store prediction results
        # This prevents re-running prediction on every Streamlit rerender
        if "last_file" not in st.session_state:
            st.session_state.last_file = None
        if "last_model" not in st.session_state:
            st.session_state.last_model = None
        if "pred_result" not in st.session_state:
            st.session_state.pred_result = None
        if "llm_result" not in st.session_state:
            st.session_state.llm_result = None
        if "all_probs" not in st.session_state:
            st.session_state.all_probs = None

        predict_clicked = st.button("🔍 Analyse Plant", use_container_width=True)

        # Only run prediction when button is clicked
        if predict_clicked:
            current_file_id = f"{uploaded_file.name}_{len(image_bytes)}_{model_choice}"
            try:
                with st.spinner("Analysing leaf tissue..."):
                    pred_label, confidence, all_probs = get_prediction(model, image_bytes, classes)
                st.session_state.pred_result = (pred_label, confidence)
                st.session_state.all_probs = all_probs
                st.session_state.last_file = current_file_id
                st.session_state.last_model = model_choice

                # Get LLM result (cached per disease name)
                with st.spinner("Generating AI explanation..."):
                    st.session_state.llm_result = ask_llm_cached(pred_label)

                # Save history
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

    else:
        # Reset when no file uploaded
        st.session_state.pred_result = None
        st.session_state.llm_result = None
        st.session_state.all_probs = None

        st.markdown("""
        <div class="upload-zone">
            <span class="upload-icon">🍃</span>
            <p class="upload-text">Upload a plant leaf image to begin</p>
            <p class="upload-hint">PNG · JPG · JPEG · BMP · TIFF supported</p>
        </div>
        """, unsafe_allow_html=True)


# --------------------------
# RIGHT COLUMN — RESULTS
# --------------------------
with col_right:
    st.markdown('<div class="card-title">🧬 Analysis Results</div>', unsafe_allow_html=True)

    if st.session_state.pred_result:
        pred_label, confidence = st.session_state.pred_result
        all_probs = st.session_state.all_probs
        healthy = is_healthy(pred_label)
        result_class = "result-healthy" if healthy else "result-disease"
        status_tag = '<span class="tag tag-healthy">✓ Healthy</span>' if healthy else '<span class="tag tag-disease">⚠ Disease Detected</span>'
        conf_pct = int(confidence * 100)
        bar_color = "#34d399" if healthy else "#fbbf24"

        st.markdown(f"""
        <div class="result-box {result_class}">
            <div style="margin-bottom:12px;">{status_tag}</div>
            <div class="result-label">{format_label(pred_label)}</div>
            <div class="result-confidence">
                <span style="font-weight:600; color:{bar_color}; font-size:1rem;">{conf_pct}%</span>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{conf_pct}%; background:linear-gradient(90deg,{bar_color},{bar_color}88);"></div>
                </div>
                <span style="font-size:0.75rem; color:#4ade8066;">confidence</span>
            </div>
            <div style="font-size:0.78rem; color:#4ade8066; margin-top:4px;">
                Raw score: {confidence:.4f} &nbsp;|&nbsp; Model: {model_choice}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top 3 predictions
        if show_top3 and all_probs is not None:
            st.markdown('<div class="card-title" style="margin-top:20px;">📊 Top Predictions</div>', unsafe_allow_html=True)
            top3_idx = all_probs.argsort()[-3:][::-1]
            for rank, idx in enumerate(top3_idx):
                label = classes[idx]
                prob = all_probs[idx]
                pct = int(prob * 100)
                is_h = is_healthy(label)
                dot_color = "#34d399" if is_h else "#fbbf24"
                bar_w = max(pct, 2)
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:0.8rem; color:#f0fdf4;">
                            {'🥇' if rank==0 else '🥈' if rank==1 else '🥉'} {format_label(label)}
                        </span>
                        <span style="font-size:0.8rem; font-weight:600; color:{dot_color};">{pct}%</span>
                    </div>
                    <div style="height:4px; background:#1f3a1f; border-radius:2px; overflow:hidden;">
                        <div style="height:100%; width:{bar_w}%; background:{dot_color}; border-radius:2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # LLM Explanation
        if st.session_state.llm_result:
            st.markdown('<div class="llm-section">', unsafe_allow_html=True)
            st.markdown('<div class="llm-title">🧠 AI Expert Explanation</div>', unsafe_allow_html=True)
            st.markdown(st.session_state.llm_result)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div class="card" style="text-align:center; padding:60px 24px;">
            <div style="font-size:3rem; margin-bottom:16px;">🔬</div>
            <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
                        color:#4ade8044; letter-spacing:0.05em;">
                Awaiting Analysis
            </div>
            <div style="font-size:0.8rem; color:#4ade8033; margin-top:8px;">
                Upload an image and click Analyse Plant
            </div>
        </div>
        """, unsafe_allow_html=True)


# --------------------------
# HISTORY SECTION
# --------------------------
st.markdown("<hr>", unsafe_allow_html=True)

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="card-title">📊 Prediction History</div>', unsafe_allow_html=True)
with col_h2:
    if st.button("🗑 Clear History", use_container_width=True):
        if os.path.exists("results.csv"):
            os.remove("results.csv")
            st.success("History cleared.")
            st.rerun()

results_path = "results.csv"
if os.path.exists(results_path):
    try:
        df_hist = pd.read_csv(results_path)
        df_display = df_hist.tail(20).copy()
        df_display["confidence"] = df_display["confidence"].apply(lambda x: f"{x:.1%}")
        df_display["status"] = df_display["prediction"].apply(
            lambda x: "✅ Healthy" if "healthy" in x.lower() else "⚠️ Diseased"
        )
        df_display["prediction"] = df_display["prediction"].apply(format_label)
        df_display = df_display[["timestamp", "file_name", "model", "prediction", "confidence", "status"]]
        df_display.columns = ["Timestamp", "File", "Model", "Prediction", "Confidence", "Status"]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Download button
        csv_data = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Full History (CSV)",
            data=csv_data,
            file_name="phytoscan_history.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Could not read results.csv: {e}")
else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:32px;">
        <div style="color:#4ade8044; font-size:0.85rem;">
            No prediction history yet — run your first analysis above.
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# FOOTER
# --------------------------
st.markdown("""
<div style="text-align:center; padding:32px 0 16px 0; border-top:1px solid #1f3a1f; margin-top:32px;">
    <div style="font-family:'Syne',sans-serif; font-size:0.8rem; color:#4ade8033; letter-spacing:0.1em;">
        PHYTOSCAN &nbsp;·&nbsp; AI PLANT DISEASE DETECTION &nbsp;·&nbsp; BUILT WITH PYTORCH + STREAMLIT
    </div>
</div>
""", unsafe_allow_html=True)
