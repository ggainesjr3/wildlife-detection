from ultralytics import YOLO
import os

# Initialize with YOLOv8 nano (High speed, low overhead for rosencrantz)
model = YOLO('yolov8n.pt') 

data_yaml = '/home/gary/wildlife-detection/wildlife.yaml'

if __name__ == "__main__":
    if not os.path.exists(data_yaml):
        print(f"❌ Error: {data_yaml} not found.")
    else:
        print("🚀 Starting Wildlife Detection Training...")
        results = model.train(
            data=data_yaml,
            epochs=50,
            imgsz=640,
            batch=16,
            name='serengeti_v1'
        )
        print("✅ Training complete. Model saved in runs/detect/serengeti_v1/weights/best.pt")
