import streamlit as st
from PIL import Image

# 1) Page setup
st.set_page_config(page_title="LeafGuard", layout="centered")

# 2) Title + instructions
st.title("🍅 Tomato Leaf Classifier")
st.write("Upload a leaf image and get a disease prediction.")

# 3) File uploader
img_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

# 4) Display the uploaded image
if img_file:
    img = Image.open(img_file)
    st.image(img, caption="Your upload", use_column_width=True)
    
    # 5) Stub for prediction result
    st.write("Prediction: _(model output will go here)_")
