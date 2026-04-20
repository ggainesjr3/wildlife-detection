import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# 1. Page Configuration
st.set_page_config(page_title="Wildlife Bouncer", page_icon="🐾")
st.title("🐾 Wildlife Detection Dashboard")

# 2. Path Check - Updated for Cloud Deployment
# This looks for 'best.pt' in the same folder as app.py
MODEL_PATH = "best.pt"

# 3. Loading with Status Indicators
if not os.path.exists(MODEL_PATH):
    st.error(f"🚨 Model weights ('{MODEL_PATH}') not found in root directory. Please ensure the file is uploaded to GitHub.")
else:
    @st.cache_resource
    def load_model():
        # Loads the YOLO model into memory once
        return YOLO(MODEL_PATH)

    with st.spinner("Loading the Bouncer's brain..."):
        model = load_model()
    st.sidebar.success("Model Loaded Successfully")

    # 4. The Upload Station
    uploaded_file = st.file_uploader("Upload a Serengeti snapshot...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Target Image", use_container_width=True)
        
        if st.button("🔍 Run Inspection"):
            with st.spinner("Analyzing pixels..."):
                results = model(img)
                # Plot the results (draws bounding boxes)
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Detection Results", use_container_width=True)
                
                # 5. Analysis Report
                st.write("### Analysis Report:")
                for result in results:
                    if len(result.boxes) == 0:
                        st.info("No animals detected in this frame.")
                    else:
                        for box in result.boxes:
                            label = model.names[int(box.cls[0])]
                            conf = box.conf[0]
                            st.write(f"- Identified **{label}** ({conf:.2%} confidence)")

st.sidebar.info("This project uses YOLOv8 Nano for real-time wildlife identification.")