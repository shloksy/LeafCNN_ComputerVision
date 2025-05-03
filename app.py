import streamlit as st
st.set_page_config(page_title="LeafGuard", layout="centered")

from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

from model import LeafDiseaseClassifier

@st.cache(allow_output_mutation=True)
def load_model(checkpoint_path: str):
    model = LeafDiseaseClassifier()
    ckpt   = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

model = load_model("checkpoint_epoch_19.pth")

MEAN = [0.4516, 0.4654, 0.4073]
STD = [0.1550, 0.1325, 0.1726]

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Page setup
st.title("🌿 LeafGuard")
st.write("Upload a tomato leaf image and get a disease prediction.")

class_names = ['Bacterial_Spot', 'Early_Blight', 
               'Healthy', 'Late_Blight', 'Leaf_Mold', 
               'Mosaic_Virus', 'Septoria_Leaf_Spot', 
               'Target_Spot', 'Two_Spotted_Spider_Mite', 
               'Yellowleaf__Curl_Virus']

uploaded = st.file_uploader("Choose an image file", type=["jpg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        prediction = F.softmax(output, dim=1)
        conf, idx = prediction.max(dim=1)
        pred_label = class_names[idx.item()]
        conf_pct   = conf.item() * 100

    st.success(f"**Prediction:** {pred_label} ({conf_pct:.2f}%)")
