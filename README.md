# 📂 SYSTEM_ARCH: SERENGETI_OBJECT_DETECTOR_V1
# STATUS: [OPERATIONAL] | MODEL: [YOLOv8n] | ENVIRONMENT: [ROSENCRANTZ]
# DEVELOPER: GARY EDWARD GAINES, JR.

---

## 🛠 TACTICAL OVERVIEW
A real-time Computer Vision (CV) system trained to detect and localize wildlife in the Snapshot Serengeti dataset. Utilizing the **YOLO (You Only Look Once)** architecture, this model performs single-pass inference to identify species under varying lighting and camouflage conditions.

### 🧠 REFINED LOGIC (CV ARCHITECTURE)
Object detection requires balancing spatial localization with semantic classification:
* **[SPATIAL_LOCALIZATION]:** Predictions are mapped to a grid system using normalized bounding boxes $[x_{center}, y_{center}, w, h]$, ensuring accuracy remains invariant to image resolution.
* **[CONFIDENCE_THRESHOLDING]:** Implemented a 0.25 NMS (Non-Maximum Suppression) floor to eliminate duplicate detections and background noise.
* **[DATA_REALIGNMENT]:** Synced class mapping to ensure consistent identification across 5 core species (Zebra, Elephant, Lion, Giraffe, Cheetah).

### 🛡 DEFENSIVE PATTERNS
* **BACKGROUND_CALIBRATION:** Included a subset of "empty" frames (Negative Samples) to reduce False Positives caused by moving foliage or shadows.
* **VALIDATION_INTEGRITY:** Enforced a strict Training/Validation split to ensure the model generalizes to new environments rather than memorizing training frames.
* **NORMALIZED_SANITIZATION:** All label data was audited to ensure coordinates were relative (0 to 1), making the pipeline robust against varying sensor hardware.

---

## 🚀 DEPLOYMENT_LOGS

### INSTALL_DEPENDS
```bash
pip install ultralytics opencv-python matplotlib

EXECUTE_TRAIN
Bash

# Calibrates YOLO weights and saves best.pt
python3 src/train.py

INFERENCE_TEST
Bash

# Run detection on a new image
yolo predict model=runs/detect/serengeti_v1/weights/best.pt source='path/to/test_image.jpg'

🎙 PHILOSOPHY

    "In the Serengeti, everything is hidden until it's not. This system is designed to be the digital eyes that never blink, spotting the predator before it breaks cover."

👤 DEVELOPER_INFO

    Lead Engineer: Gary Edward Gaines, Jr.

    Focus: Computer Vision, YOLO Deployment

    Location: Philadelphia, PA / Southern NJ Area

    Host_Machine: rosencrantz
