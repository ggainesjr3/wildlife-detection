from ultralytics import YOLO
import os

def train_wildlife_bouncer():
    """
    Analogy: The First Shift.
    We take a 'Pre-trained' bouncer (yolov8n) and teach them our specific guests.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(BASE_DIR, "wildlife_data.yaml")
    
    print("--- 🏋️ Starting Training Shift ---")

    # 1. Load a pre-trained 'Nano' model (Smallest/Fastest for Edge Devices)
    # 'yolov8n.pt' is like a bouncer who knows how to see shapes but not our animals yet.
    model = YOLO('yolov8n.pt')

    # 2. Train the model
    # epochs=3: We let the bouncer walk the floor 3 times.
    # imgsz=400: The size of our mock images.
    results = model.train(
        data=CONFIG_PATH, 
        epochs=3, 
        imgsz=400, 
        plots=True
    )

    print("\n--- ✅ Training Complete! ---")
    print(f"Model saved to: {results.save_dir}")

if __name__ == "__main__":
    train_wildlife_bouncer()