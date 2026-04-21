import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# --------------------------
# CNN MODEL ARCHITECTURE
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
# UTILS
# --------------------------

def load_cnn_model(model_path, num_classes):
    model = PlantDiseaseCNN(num_classes)
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint
    model.eval()
    return model

def predict_disease(model, image_bytes, classes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return classes[predicted.item()], confidence.item()

# --------------------------
# RAG SETUP
# --------------------------

def setup_rag(knowledge_base_path):
    with open(knowledge_base_path, 'r') as f:
        text = f.read()
    
    # Split by double newlines (assuming each disease is a block)
    docs = text.split('\n\n')
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(docs, embeddings)
    return vectorstore

def get_rag_context(vectorstore, query, k=2):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

# --------------------------
# LLM SETUP
# --------------------------

def get_groq_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
