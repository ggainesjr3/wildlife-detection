import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="Wildlife Bouncer", page_icon="🐾")
st.title("🐾 Wildlife Detection Dashboard")

# 1. Path Check
MODEL_PATH = os.path.join(os.getcwd(), "runs/detect/train/weights/best.pt")

# 2. Loading with Status Indicators
if not os.path.exists(MODEL_PATH):
    st.error(f"🚨 Model weights not found at {MODEL_PATH}. Please run your training script first!")
else:
    @st.cache_resource
    def load_model():
        # This will only run once and stay in memory
        return YOLO(MODEL_PATH)

    with st.spinner("Loading the Bouncer's brain..."):
        model = load_model()
    st.sidebar.success("Model Loaded Successfully")

    # 3. The Upload Station
    uploaded_file = st.file_uploader("Upload a Serengeti snapshot...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Target Image", use_container_width=True)
        
        if st.button("🔍 Run Inspection"):
            with st.spinner("Analyzing pixels..."):
                results = model(img)
                # Plot the results
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Detection Results", use_container_width=True)
                
                # Show specific counts
                st.write("### Analysis Report:")
                for result in results:
                    if len(result.boxes) == 0:
                        st.info("No animals detected in this frame.")
                    else:
                        for box in result.boxes:
                            label = model.names[int(box.cls[0])]
                            conf = box.conf[0]
                            st.write(f"- Identified **{label}** ({conf:.2%} confidence)")