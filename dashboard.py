import streamlit as st
import pandas as pd
import sqlite3
from datetime import date
import plotly.express as px
import os
import cv2
import numpy as np
import tensorflow as tf

# --- Configuration ---
DB_FILE = "activity.db"
MODEL_PATH = "autoencoder.h5"
CAPTURE_INTERVAL_SECONDS = 10
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# --- Database and Data Loading ---
def load_data_for_date(selected_date):
    """Query the database for all logs on a specific date."""
    try:
        conn = sqlite3.connect(DB_FILE)
        query = "SELECT * FROM logs WHERE DATE(timestamp) = ?"
        df = pd.read_sql_query(query, conn, params=(selected_date.strftime('%Y-%m-%d'),))
        conn.close()
        return df
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

# --- Classification ---
def classify_activity(row):
    """Classifies a single activity record based on predefined rules."""
    app_name = row['active_app_name'].lower()
    window_title = row['active_window_title'].lower()
    # ocr_text is not used in the current classification but kept for future use
    # ocr_text = str(row['ocr_text']).lower() if pd.notna(row['ocr_text']) else ""
    if any(keyword in app_name for keyword in ["code.exe", "visual studio code", "sublime_text"]):
        return "Coding"
    if any(keyword in window_title for keyword in ["outlook", "gmail"]):
        return "Email"
    if any(keyword in window_title for keyword in ["youtube", "netflix"]):
        return "Entertainment"
    return "General"

# --- Anomaly Detection ---
@st.cache_resource
def load_model(path):
    """Load the trained autoencoder model."""
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

def preprocess_image(path):
    """Load and preprocess a single image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img.astype('float32') / 255.0
    return img.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

def find_top_anomalies(model, df):
    """Find the top 5 anomalies based on reconstruction error."""
    errors = []
    for i, row in df.iterrows():
        image_path = row['screenshot_path']
        if not os.path.exists(image_path):
            errors.append(0)
            continue
        
        original_img = preprocess_image(image_path)
        reconstructed_img = model.predict(original_img, verbose=0)
        mse = np.mean(np.square(original_img - reconstructed_img))
        errors.append(mse)
    
    df['reconstruction_error'] = errors
    return df.nlargest(5, 'reconstruction_error')

# --- Main Dashboard ---
def main():
    st.set_page_config(layout="wide")
    st.title("Ambient Contextual AI Productivity Dashboard")

    autoencoder = load_model(MODEL_PATH)
    
    st_date = st.date_input("Select a date to review", date.today())
    
    if st_date:
        st.header(f"Activity for {st_date.strftime('%Y-%m-%d')}")
        daily_data = load_data_for_date(st_date)
        
        if not daily_data.empty:
            daily_data['category'] = daily_data.apply(classify_activity, axis=1)
            daily_data['timestamp'] = pd.to_datetime(daily_data['timestamp'])

            # --- Visualizations ---
            st.subheader("Productivity Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                time_spent = daily_data['category'].value_counts() * CAPTURE_INTERVAL_SECONDS / 60
                fig_pie = px.pie(time_spent.reset_index(), values='count', names='category', title="Time Spent per Activity (Minutes)")
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                # Timeline Chart - FIX
                timeline_df = daily_data.set_index('timestamp').resample('10T')['category'].apply(lambda x: x.mode()[0] if not x.empty else None).dropna().reset_index()
                timeline_df.rename(columns={'timestamp': 'time_block'}, inplace=True)

                fig_timeline = px.bar(timeline_df, x='time_block', y='category', color='category',
                                      title="Activity Timeline (10-Minute Intervals)",
                                      labels={'time_block': 'Time', 'category': 'Activity'})
                fig_timeline.update_layout(yaxis_title=None)
                st.plotly_chart(fig_timeline, use_container_width=True)

            # --- Anomaly Report ---
            if autoencoder is not None:
                st.subheader("Anomaly Report")
                with st.spinner("Analyzing screenshots for anomalies..."):
                    top_anomalies = find_top_anomalies(autoencoder, daily_data.copy())
                
                st.write("These are the moments from your day that least match your typical activity patterns.")
                for i, row in top_anomalies.iterrows():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(row['screenshot_path'], caption=f"Timestamp: {row['timestamp']}")
                    with col2:
                        st.warning(f"**High Anomaly Detected at {row['timestamp'].strftime('%H:%M:%S')}**")
                        st.write(f"**Activity:** {row['active_window_title']}")
                        st.write(f"**Category:** {row['category']}")
                        st.metric("Reconstruction Error (MSE)", f"{row['reconstruction_error']:.6f}")
            else:
                st.warning("`autoencoder.h5` not found. Please train the model first (`autoencoder_trainer.py`).")

            with st.expander("View Raw Data"):
                st.dataframe(daily_data)
        else:
            st.warning("No activity was logged on this date.")

if __name__ == "__main__":
    main()
