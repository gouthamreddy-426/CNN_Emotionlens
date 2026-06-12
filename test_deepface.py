from deepface import DeepFace
import numpy as np

# Create a dummy image
img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
try:
    res = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    print("DeepFace works!")
    print(res)
except Exception as e:
    print("DeepFace error:", e)
