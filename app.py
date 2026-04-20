import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter

# 1. Page Configuration
st.set_page_config(
    page_title="Wildlife Bouncer | Gary Gaines", 
    page_icon="🐾",
    layout="wide"
)

# 2. Sidebar Branding
with st.sidebar:
    st.image("https://img.icons8.com/color/96/safari.png", width=100)
    st.title("Wildlife Bouncer AI")
    st.markdown("""
    **Developer:** Gary Edward Gaines, Jr.  
    **Model:** YOLOv8 Nano  
    ---
    [GitHub Profile](https://github.com/ggainesjr3)  
    """)

# 3. Main Interface
st.title("🐾 Wildlife Detection Dashboard")

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("🚨 Model weights not found. Please ensure 'best.pt' is in the root directory.")
else:
    @st.cache_resource
    def load_model():
        return YOLO(MODEL_PATH)

    model = load_model()
    uploaded_file = st.file_uploader("Upload a Serengeti snapshot...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(img, use_container_width=True)
            
        if st.button("🔍 Run ML Inspection", use_container_width=True):
            results = model(img)
            res_plotted = results[0].plot()
            
            with col2:
                st.subheader("Detection Results")
                st.image(res_plotted, use_container_width=True)
            
            st.write("---")
            st.subheader("📊 Species Tally")
            labels_found = [model.names[int(box.cls[0])] for result in results for box in result.boxes]
            counts = Counter(labels_found)
            
            if counts:
                cols = st.columns(len(counts))
                for i, (species, count) in enumerate(counts.items()):
                    cols[i].metric(label=species.capitalize(), value=count)