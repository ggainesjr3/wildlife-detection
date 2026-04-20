from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def test_the_bouncer():
    """
    Analogy: The Surprise Inspection.
    We create a brand new 'guest' to see if the bouncer recognizes them.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Model weights not found! Did you run train_model.py first?")
        return

    # 1. Load the 'Trained Brain'
    model = YOLO(MODEL_PATH)

    # 2. Create a NEW 'unseen' image (a fake Zebra)
    # 400x400 gray image with a white square in a NEW location
    test_img = np.full((400, 400, 3), 50, dtype=np.uint8)
    cv2.rectangle(test_img, (250, 250), (350, 350), (255, 255, 255), -1)
    
    test_path = os.path.join(BASE_DIR, "test_zebra.jpg")
    cv2.imwrite(test_path, test_img)

    # 3. Run Detection (The Bouncer's Decision)
    results = model(test_path)

    # 4. Show the Results
    # YOLO has a built-in 'plot' function to show boxes and labels
    res_plotted = results[0].plot()
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(res_rgb)
    plt.title("Bouncer Decision: Animal Identified!")
    plt.axis('off')
    plt.show()
    
    print(f"✅ Inspection Complete! Look at the pop-up to see the 'Zebra' tag.")

if __name__ == "__main__":
    test_the_bouncer()