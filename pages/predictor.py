import streamlit as st
import pandas as pd
from PIL import Image
from utils.prediction import predict_emotion
import plotly.express as px

st.title("🎯 Emotion Prediction")
st.markdown("Provide a close-up image of a face to detect the emotion.")

col1, col2 = st.columns([1, 1])

with col1:
    input_method = st.radio("Choose Input Method:", ["Upload Image", "Take Photo"], horizontal=True)

    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Face Image",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(
                    image,
                    caption="Uploaded Image",
                    use_container_width=True
                )
            except Exception as e:
                st.error("Error opening the image. Please upload a valid image file.")
    elif input_method == "Take Photo":
        camera_photo = st.camera_input("Take a picture")
        if camera_photo is not None:
            try:
                image = Image.open(camera_photo)
            except Exception as e:
                st.error("Error processing camera image.")

with col2:
    if image is not None:
        with st.spinner("Analyzing emotion..."):
            emotion, confidence, predictions = predict_emotion(image)

        if emotion is None:
            st.error("An error occurred during prediction. Please try another image.")
        else:
            st.success(f"**Predicted Emotion:** {emotion}")
            st.info(f"**Confidence:** {confidence*100:.2f}%")

            emotions = [
                "Angry", "Disgust", "Fear", "Happy",
                "Neutral", "Sad", "Surprise"
            ]

            df = pd.DataFrame({
                "Emotion": emotions,
                "Probability": [p * 100 for p in predictions]
            })

            st.subheader("Prediction Probabilities")

            fig = px.bar(
                df, x="Emotion", y="Probability",
                text=[f"{p:.1f}%" for p in df["Probability"]],
                title="Confidence Levels per Emotion"
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)