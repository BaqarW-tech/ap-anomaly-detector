import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

# ----------------------------
# App Header
# ----------------------------
st.set_page_config(page_title="AP Anomaly Detector", layout="wide")
st.title("AP Anomaly Detector")
st.markdown(
    "Upload your dataset and detect anomalies using Isolation Forest."
)

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File loaded successfully!")

        # ----------------------------
        # Data Cleaning
        # ----------------------------
        numeric_cols = ["Clicks", "Impressions", "CTR", "Position"]
        for col in numeric_cols:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                st.stop()

        # Clean CTR column if it contains '%'
        df["CTR"] = df["CTR"].astype(str).str.replace("%", "").astype(float) / 100

        # Drop rows with NaNs
        df = df.dropna(subset=numeric_cols)

        # ----------------------------
        # Model Training
        # ----------------------------
        X = df[numeric_cols]
        model = IsolationForest(contamination=0.05, random_state=42)
        df["Anomaly"] = model.fit_predict(X)
        df["Anomaly_Label"] = df["Anomaly"].map({1: "Normal", -1: "Anomaly"})

        # ----------------------------
        # Data Preview
        # ----------------------------
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # ----------------------------
        # Anomaly Visualization
        # ----------------------------
        st.subheader("Anomaly Scatter Plot")
        fig = px.scatter(
            df,
            x="Impressions",
            y="Clicks",
            color="Anomaly_Label",
            hover_data=numeric_cols,
            title="Impressions vs Clicks with Anomalies"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"Detected {df['Anomaly_Label'].value_counts().get('Anomaly',0)} anomalies out of {len(df)} rows."
        )

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload a CSV file to begin anomaly detection.")