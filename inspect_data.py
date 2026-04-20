import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

def inspect_wildlife_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    METADATA = os.path.join(BASE_DIR, "data", "annotations.csv")
    IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
    
    if not os.path.exists(METADATA):
        print("❌ No data found. Run 'python3 setup_mock_data.py' first!")
        return

    df = pd.read_csv(METADATA)
    print(f"--- 🦓 Portfolio Mock Inspection ---")
    print(f"Found {len(df)} animals in the guest list.\n")

    for _, row in df.iterrows():
        img_path = os.path.join(IMAGE_DIR, f"{row['capture_id']}.jpg")
        
        # Load the image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw the bounding box (This is the 'Bouncer' labeling the guest)
        cv2.rectangle(image, 
                      (row['x_min'], row['y_min']), 
                      (row['x_min']+row['width'], row['y_min']+row['height']), 
                      (0, 255, 0), 2)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"Label: {row['species']} | ID: {row['capture_id']}")
        plt.axis('off')
        plt.show()
        print(f"Verified: {row['species']}")

if __name__ == "__main__":
    inspect_wildlife_data()