import streamlit as st
st.set_page_config(page_title="LeafGuard", layout="centered")

import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

from model import LeafDiseaseClassifier
from disease_descriptions import descriptions

@st.cache_resource
def load_model(checkpoint_path: str):
    model = LeafDiseaseClassifier()
    ckpt   = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def predict(img):
    col1, col2, col3 = st.columns([1, 2, 1])
    col2.image(img, caption="Input Image", width=300)
    
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        prediction = F.softmax(output, dim=1)
        conf, idx = prediction.max(dim=1)
        pred_label = class_names[idx.item()]
        conf_pct   = conf.item() * 100

    st.markdown(
    f"<span style='color: green; font-size:26px;'>"
    f"Prediction: {pred_label}<br>"
    f"Confidence: {conf_pct:.2f}%"
    "</span>",
    unsafe_allow_html=True)

    st.write(descriptions[idx.item()])

model = load_model("checkpoint_epoch_19.pth")

MEAN = [0.4516, 0.4654, 0.4073]
STD = [0.1550, 0.1325, 0.1726]

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Page setup
st.title("🌿 Leaf Guard")
st.write("Upload a tomato leaf image and get a disease prediction.")

class_names = ['Bacterial Spot', 'Early Blight', 
               'Healthy', 'Late Blight', 'Leaf Mold', 
               'Mosaic Virus', 'Septoria Leaf_Spot', 
               'Target Spot', 'Two Spotted Spider Mite', 
               'Yellowleaf Curl Virus']

SAMPLE_DIR = "samples"
sample_files = sorted([f for f in os.listdir(SAMPLE_DIR)], key=lambda s: s.lower())

uploaded = st.file_uploader("Choose an image file")

if not uploaded:
    choice = st.selectbox(
        "Or try a sample image!", 
        [""] + sample_files,
        format_func=lambda x: " " if x=="" else x
    )
    if choice:
        img_path = os.path.join(SAMPLE_DIR, choice)
        img = Image.open(img_path).convert("RGB")
    else: st.stop()
    
    predict(img)
    
else:
    img = Image.open(uploaded).convert("RGB")
    
    predict(img)
