import streamlit as st
import os
import pandas as pd
import plotly.express as px

st.title("📈 Analytics")

train_path = "dataset/train"

if not os.path.exists(train_path):
    st.warning("Training dataset not found. Please place images in 'dataset/train'.")
else:
    data = []

    for emotion in os.listdir(train_path):
        emotion_folder = os.path.join(train_path, emotion)
        
        if os.path.isdir(emotion_folder):
            count = len(os.listdir(emotion_folder))
            data.append([emotion, count])

    if not data:
        st.warning("No data found in the training dataset.")
    else:
        df = pd.DataFrame(data, columns=["Emotion", "Images"])

        fig = px.pie(
            df,
            values="Images",
            names="Emotion",
            title="Emotion Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            df,
            x="Emotion",
            y="Images",
            title="Images Per Emotion",
            text="Images",
            color="Emotion"
        )
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("View Raw Data"):
            st.dataframe(df, use_container_width=True)