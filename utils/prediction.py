import os
import numpy as np
import cv2
import streamlit as st
from deepface import DeepFace

class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

def predict_emotion(image):
    try:
        # Ensure image is RGB
        if hasattr(image, "convert"):
            image = image.convert("RGB")
            
        image_np = np.array(image)
        
        # DeepFace analyzes the image and detects emotion automatically
        result = DeepFace.analyze(image_np, actions=['emotion'], enforce_detection=False)
        
        if isinstance(result, list):
            res = result[0]
        else:
            res = result
            
        # Extract dominant emotion and capitalize it to match our UI classes
        emotion = res['dominant_emotion'].capitalize()
        
        # DeepFace returns probabilities out of 100
        predictions_dict = res['emotion']
        
        predictions_list = []
        for c in class_names:
            prob = predictions_dict.get(c.lower(), 0.0)
            predictions_list.append(prob / 100.0)
            
        confidence = float(predictions_dict.get(emotion.lower(), 0.0)) / 100.0
        
        return emotion, confidence, predictions_list
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0, None