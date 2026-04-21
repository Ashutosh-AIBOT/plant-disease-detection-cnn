import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import io
import os
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import torch.nn.functional as F
from utils.app_utils import load_cnn_model, predict_disease, setup_rag, get_rag_context, get_groq_llm

# --------------------------
# CONFIG + ENV
# --------------------------
load_dotenv()
st.set_page_config(
    page_title="PhytoScan — Plant Disease AI",
    page_icon="🌿",
    layout="wide"
)

# --------------------------
# PREMIUM CSS (Dashboard + Chat)
# --------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0f0a;
    --bg-secondary: #111811;
    --accent-green: #4ade80;
    --text-primary: #f0fdf4;
    --border: #1f3a1f;
}

.stApp { background: var(--bg-primary); color: var(--text-primary); font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--bg-secondary) !important; border-right: 1px solid var(--border) !important; }

/* Dashboard Cards */
.stat-card {
    background: #141f14; border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; text-align: center; transition: 0.3s;
}
.stat-card:hover { border-color: var(--accent-green); box-shadow: 0 0 20px #4ade8011; }
.stat-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: var(--accent-green); }
.stat-label { font-size: 0.7rem; color: #86efac88; text-transform: uppercase; margin-top: 5px; }

/* Header from app01.py */
.phyto-header {
    display: flex; align-items: center; gap: 16px; padding: 32px 0 24px 0;
    border-bottom: 1px solid var(--border); margin-bottom: 32px;
}
.phyto-logo {
    width: 52px; height: 52px; background: linear-gradient(135deg, #4ade80, #34d399);
    border-radius: 14px; display: flex; align-items: center; justify-content: center;
    font-size: 26px; box-shadow: 0 0 24px #4ade8044;
}
.phyto-title {
    font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
    color: var(--text-primary); margin: 0; line-height: 1; letter-spacing: -0.02em;
}
.phyto-subtitle {
    font-size: 0.85rem; color: #4ade8088; margin: 4px 0 0 0;
    letter-spacing: 0.08em; text-transform: uppercase; font-weight: 400;
}

/* Chat Styling */
.stChatMessage { background: #141f14 !important; border: 1px solid #1f3a1f !important; border-radius: 15px !important; margin-bottom: 10px !important; }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = "plant_disease_cnn.pth"
KNOWLEDGE_BASE = "data/knowledge_base.txt"
MAIN_CLASSES = [
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
]

# --------------------------
# SESSION STATE
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()

@st.cache_resource
def load_resources():
    model = load_cnn_model(MODEL_PATH, len(MAIN_CLASSES))
    vectorstore = setup_rag(KNOWLEDGE_BASE)
    llm = get_groq_llm()
    return model, vectorstore, llm

with st.spinner("PhytoScan AI Initializing..."):
    model, vectorstore, llm = load_resources()

# --------------------------
# SIDEBAR
# --------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:800;
                    color:#4ade80; letter-spacing:-0.01em;">🌿 PhytoScan</div>
        <div style="font-size:0.72rem; color:#4ade8066; text-transform:uppercase;
                    letter-spacing:0.12em; margin-top:2px;">AI-Powered Plant Disease Detection System</div>
    </div>
    """, unsafe_allow_html=True)
    
    action_mode = st.radio("🛠 SELECT ACTION", ["💬 CHAT & DIAGNOSTIC", "📁 UPLOAD LEAF", "📷 LIVE SCAN"])
    
    if st.button("🗑 RESET SESSION", use_container_width=True):
        st.session_state.messages = []
        st.session_state.processed_ids = set()
        st.rerun()

# --------------------------
# DASHBOARD HEADER
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
def get_stats():
    if not os.path.exists("results.csv"): return 0, 0
    df = pd.read_csv("results.csv")
    return len(df), df["prediction"].str.contains("healthy", case=False).sum()

total, healthy = get_stats()
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f'<div class="stat-card"><div class="stat-value">{total}</div><div class="stat-label">Total Scans</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="stat-card"><div class="stat-value">{healthy}</div><div class="stat-label">Healthy Plants</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="stat-card"><div class="stat-value">{total-healthy}</div><div class="stat-label">Issues Detected</div></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="stat-card"><div class="stat-value">38</div><div class="stat-label">AI Classes</div></div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------
# LOGIC: IMAGE HANDLING
# --------------------------
def handle_diagnostic(image_file, file_id):
    if file_id in st.session_state.processed_ids:
        return
    
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    st.session_state.messages.append({"role": "user", "content": "Scanning this leaf...", "image": image})
    
    with st.spinner("Running AI Diagnostic..."):
        label, conf = predict_disease(model, image_bytes, MAIN_CLASSES)
    
    if conf < 0.45:
        response = "⚠️ **Result:** No match found in our database model. The image might be too blurry or is an unsupported variety."
    else:
        with st.spinner("Consulting knowledge base..."):
            context = get_rag_context(vectorstore, label)
            prompt = f"User uploaded a plant leaf image. Model detected: {label} ({conf*100:.1f}% confidence).\nContext: {context}\nProvide a professional diagnostic report."
            ai_resp = llm.invoke(prompt)
            response = f"### Diagnosis: {label.replace('___',' — ').replace('_',' ')}\n\n" + ai_resp.content
            # Save history
            pd.DataFrame([{"timestamp": time.ctime(), "prediction": label, "confidence": conf}]).to_csv("results.csv", mode='a', header=not os.path.exists("results.csv"), index=False)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.processed_ids.add(file_id)
    st.rerun()

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg: st.image(msg["image"], width=400)

# Inputs
if action_mode == "📁 UPLOAD LEAF":
    up_file = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if up_file: handle_diagnostic(up_file, f"{up_file.name}_{up_file.size}")

elif action_mode == "📷 LIVE SCAN":
    cam_file = st.camera_input("Capture leaf image", label_visibility="collapsed")
    if cam_file: handle_diagnostic(cam_file, f"cam_{int(time.time())}")

# Chat input
if chat_prompt := st.chat_input("Ask about treatment or plant care..."):
    st.session_state.messages.append({"role": "user", "content": chat_prompt})
    with st.chat_message("user"): st.markdown(chat_prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history = "\n".join([m["content"][:200] for m in st.session_state.messages[-3:]])
            rag = get_rag_context(vectorstore, chat_prompt)
            full_prompt = f"User asks: {chat_prompt}\nHistory:\n{history}\n\nData:\n{rag}\n\nAssistant:"
            res = llm.invoke(full_prompt)
            st.markdown(res.content)
            st.session_state.messages.append({"role": "assistant", "content": res.content})
