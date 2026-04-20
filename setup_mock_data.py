import pandas as pd
import cv2
import numpy as np
import os

def create_mock_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMG_DIR = os.path.join(DATA_DIR, "images")
    
    # Create directories if they don't exist
    os.makedirs(IMG_DIR, exist_ok=True)

    # 1. Create a 'Mock Guest List'
    # We define: CaptureID, Species, and fake Bounding Box coordinates
    mock_data = {
        'capture_id': ['zebra_1', 'lion_1', 'elephant_1'],
        'species': ['zebra', 'lion', 'elephant'],
        'x_min': [50, 150, 200],
        'y_min': [50, 100, 50],
        'width': [100, 80, 150],
        'height': [100, 80, 200]
    }
    df = pd.DataFrame(mock_data)
    df.to_csv(os.path.join(DATA_DIR, "annotations.csv"), index=False)
    print("✅ Created: data/annotations.csv")

    # 2. Create 'Mock Bottles' (Images with colored squares)
    colors = [(255, 255, 255), (0, 255, 255), (255, 100, 0)] # White, Yellow, Orange
    
    for i, row in df.iterrows():
        # Create a 400x400 gray image
        img = np.full((400, 400, 3), 50, dtype=np.uint8)
        
        # Draw a colored square where the 'animal' is
        cv2.rectangle(img, 
                      (row['x_min'], row['y_min']), 
                      (row['x_min']+row['width'], row['y_min']+row['height']), 
                      colors[i], -1)
        
        save_path = os.path.join(IMG_DIR, f"{row['capture_id']}.jpg")
        cv2.imwrite(save_path, img)
        print(f"✅ Created Mock Image: {save_path}")

if __name__ == "__main__":
    create_mock_assets()