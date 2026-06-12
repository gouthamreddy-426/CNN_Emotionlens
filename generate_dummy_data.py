import os
import numpy as np
from PIL import Image

emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
base_dir = "dataset"

def create_dummy_images(split, num_images_per_class=10):
    for emotion in emotions:
        folder_path = os.path.join(base_dir, split, emotion)
        os.makedirs(folder_path, exist_ok=True)
        
        for i in range(num_images_per_class):
            # Create a 48x48 random grayscale image
            random_pixels = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
            img = Image.fromarray(random_pixels, mode='L')
            img.save(os.path.join(folder_path, f"dummy_{i}.jpg"))

print("Generating dummy dataset for testing...")
create_dummy_images("train", 20)  # 20 images per emotion for training
create_dummy_images("test", 5)    # 5 images per emotion for testing
print("Dummy dataset generated successfully! You can now run `python train_model.py`")
