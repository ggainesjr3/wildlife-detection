import pandas as pd
import os

def normalize_to_yolo():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "data", "annotations.csv")
    LABEL_DIR = os.path.join(BASE_DIR, "data", "labels")
    
    os.makedirs(LABEL_DIR, exist_ok=True)
    
    # The Bar's VIP List (Mapping names to numbers)
    class_map = {"zebra": 0, "lion": 1, "elephant": 2}
    
    df = pd.read_csv(CSV_PATH)
    img_width, img_height = 400, 400 # Based on our mock images

    print("--- 🏷️ Converting Labels to YOLO Format ---")
    
    for _, row in df.iterrows():
        # 1. Calculate Center X and Center Y
        # (x_min + width/2) / total_width
        x_center = (row['x_min'] + (row['width'] / 2)) / img_width
        y_center = (row['y_min'] + (row['height'] / 2)) / img_height
        
        # 2. Normalize Width and Height
        w_norm = row['width'] / img_width
        h_norm = row['height'] / img_height
        
        class_id = class_map[row['species']]
        
        # 3. Save to a text file with the SAME NAME as the image
        label_filename = f"{row['capture_id']}.txt"
        label_path = os.path.join(LABEL_DIR, label_filename)
        
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        print(f"✅ Generated Label: {label_filename}")

if __name__ == "__main__":
    normalize_to_yolo()