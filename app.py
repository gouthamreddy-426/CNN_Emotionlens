import streamlit as st

st.set_page_config(
    page_title="EmotionLens AI",
    page_icon="😊",
    layout="wide"
)

st.title("😊 EmotionLens AI")

st.subheader(
    "Facial Emotion Detection using CNN"
)

st.markdown("---")

st.write("""
EmotionLens AI detects human emotions
from facial images using a Convolutional Neural Network.

Supported Emotions:

• Angry
• Disgust
• Fear
• Happy
• Neutral
• Sad
• Surprise
""")