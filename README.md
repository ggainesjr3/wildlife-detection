🐾 Wildlife Object Detection: Snapshot Serengeti Pipeline

# 🐾 Wildlife Detection Dashboard

**Live Demo:** [Click here to view the app](https://wildlife-detection-2zsvbkwbbjmirc5aimtstq.streamlit.app/)

## 🚀 Project Overview

This project develops a lightweight, edge-ready object detection model designed to identify wildlife in camera-trap imagery. By leveraging the **YOLOv8** architecture, this pipeline provides real-time identification of species such as Zebras, Lions, and Elephants.

This project serves as a technical case study in building a robust Machine Learning pipeline from scratch, even when faced with significant infrastructure and data access challenges.

## 🛠️ Installation & Setup

To replicate this environment and run the detection pipeline locally:

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/ggainesjr3/wildlife-detection.git](https://github.com/ggainesjr3/wildlife-detection.git)
   cd wildlife-detection

    Initialize Virtual Environment:
    Bash

    python3 -m venv venv
    source venv/bin/activate

    Install Dependencies:
    Bash

    pip install -r requirements.txt

    Generate Mock Data & Train:
    Bash

    python3 setup_mock_data.py
    python3 train_model.py

    Launch the Dashboard:
    Bash

    streamlit run app.py
   ```

🏗️ Data Engineering & The "Bridge" Strategy

A major component of this project was navigating the logistical hurdles of high-scale environmental data. I attempted to integrate three separate external datasets, but encountered the following real-world constraints:

    Snapshot Serengeti (Direct S3): Restricted access due to authentication gatekeeping (HTTP 403 Forbidden).

    DataDryad Repository: Encountered link rot and deprecated file paths (HTTP 404 Not Found).

    LILA Science (Blob Storage): Encountered permission-based connection errors and authorization blocks.

The Solution: Rather than stalling, I engineered a Synthetic Data Generation Pipeline (setup_mock_data.py). This "clean-room" approach allowed me to validate the code logic, coordinate mapping, and model architecture in a controlled environment, ensuring the system is "plug-and-play" ready for the full dataset once authenticated access is secured.
📐 Mathematical Foundation (The "ID Scanner" Logic)

Machine Learning models do not understand raw pixel locations (e.g., "50 pixels from the left"). To make the data digestible for the YOLO (You Only Look Once) architecture, I implemented Bounding Box Normalization.
Coordinate Transformation

I developed a pre-processing script (prepare_labels.py) to transform raw pixel coordinates into a scale-invariant format. For an image with width W and height H, and a bounding box defined by (xmin​,ymin​,width,height), the math is as follows:

    Center Calculation:

        xcenter​=Wxmin​+(width/2)​

        ycenter​=Hymin​+(height/2)​

    Dimension Normalization:

        wnorm​=Wwidth​

        hnorm​=Hheight​

This results in values between 0.0 and 1.0, allowing the model to detect objects regardless of the input image resolution—a critical feature for varied camera-trap hardware.
🏋️ Training & Edge Optimization

I utilized Transfer Learning with the yolov8n.pt (Nano) weights.

    The Engine: Training was performed using Stochastic Gradient Descent (SGD) to minimize the Loss Function, which balances Box Loss (spatial accuracy), Class Loss (category identification), and Objectness Loss (presence detection).

    Edge Deployment: The 'Nano' variant was chosen for its tiny ~6MB footprint. This enables high-speed inference on low-power hardware like a Raspberry Pi 4, essential for remote field deployment where power and connectivity are limited.

📊 Validation

The model's performance was validated using an Inference Script (test_model.py) and the Streamlit Dashboard (app.py) on unseen synthetic data. The model successfully identified targets in new spatial coordinates, proving it had learned generalized visual features rather than simply memorizing training positions.

Developed by Gary Edward Gaines, Jr. as a professional Machine Learning Portfolio Project.
