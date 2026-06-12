import streamlit as st
import os

st.title("📊 Dashboard")

train_path = "dataset/train"

if not os.path.exists(train_path):
    st.warning("Training dataset not found. Please place images in 'dataset/train'.")
else:
    emotion_counts = {}

    for emotion in os.listdir(train_path):
        emotion_folder = os.path.join(train_path, emotion)
        
        if os.path.isdir(emotion_folder):
            count = len(os.listdir(emotion_folder))
            emotion_counts[emotion] = count

    if not emotion_counts:
        st.warning("No data found in the training dataset.")
    else:
        st.subheader("Dataset Statistics")

        total_images = sum(emotion_counts.values())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Images", f"{total_images:,}")
        with col2:
            st.metric("Emotion Classes", len(emotion_counts))
        with col3:
            avg = total_images // len(emotion_counts) if len(emotion_counts) > 0 else 0
            st.metric("Avg. Images per Class", f"{avg:,}")

        st.subheader("Class Distribution")
        st.bar_chart(emotion_counts)