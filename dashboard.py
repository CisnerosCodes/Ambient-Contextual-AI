import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import numpy as np
from datetime import datetime, date
import plotly.express as px
from sentence_transformers import SentenceTransformer
import analysis

# --- Configuration ---
DB_FILE = "activity.db"

# --- Setup ---
st.set_page_config(page_title="Ambient Contextual AI", layout="wide")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('clip-ViT-B-32')

def load_data(selected_date):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM logs WHERE DATE(timestamp) = ?"
    df = pd.read_sql_query(query, conn, params=(selected_date.strftime('%Y-%m-%d'),))
    conn.close()
    return df

def load_summaries(selected_date):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM summaries WHERE DATE(timestamp) = ?"
    df = pd.read_sql_query(query, conn, params=(selected_date.strftime('%Y-%m-%d'),))
    conn.close()
    return df

# --- Sidebar ---
st.sidebar.title("Controls")
selected_date = st.sidebar.date_input("Select Date", date.today())

if st.sidebar.button("Set Last Screenshot as Anchor"):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT embedding_json FROM logs ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        embedding = json.loads(row[0])
        analysis.set_anchor_embedding(embedding)
        st.sidebar.success("Anchor set to last activity!")
    else:
        st.sidebar.error("No logs found to set anchor.")

if st.sidebar.button("Generate Summary Now"):
    with st.spinner("Generating summary..."):
        summary = analysis.generate_hourly_summary()
        st.sidebar.info(summary)

# --- Main Content ---
st.title("Ambient Contextual AI Dashboard")

# 1. Focus Wave
st.header("🌊 Focus Wave")
df = load_data(selected_date)
anchor_embedding = analysis.get_anchor_embedding()

if not df.empty and anchor_embedding:
    # Calculate Focus Scores
    # Handle potential None values in embedding_json
    df['embedding'] = df['embedding_json'].apply(lambda x: json.loads(x) if x else None)
    # Filter out rows where embedding is None
    df = df.dropna(subset=['embedding'])
    
    if not df.empty:
        df['focus_score'] = df['embedding'].apply(lambda x: analysis.calculate_focus_score(x, anchor_embedding))
        
        fig = px.line(df, x='timestamp', y='focus_score', title='Focus Score Over Time', range_y=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid embeddings found for this date.")
elif df.empty:
    st.info("No data for selected date.")
else:
    st.warning("Please set an Anchor State (Ideal Work State) in the sidebar to see Focus Scores.")

# 2. Narrative
st.header("📖 Daily Narrative")
summaries_df = load_summaries(selected_date)
if not summaries_df.empty:
    for index, row in summaries_df.iterrows():
        st.markdown(f"**{row['timestamp']}**: {row['summary_text']}")
else:
    st.info("No summaries generated yet.")

# 3. Semantic Search
st.header("🔍 Semantic Search")
search_query = st.text_input("Search your day (e.g., 'Reading news', 'Coding python')")

if search_query and not df.empty:
    model = load_embedding_model()
    query_embedding = model.encode(search_query)
    
    # Ensure we have embeddings
    if 'embedding' not in df.columns:
         df['embedding'] = df['embedding_json'].apply(lambda x: json.loads(x) if x else None)
         df = df.dropna(subset=['embedding'])

    if not df.empty:
        df['similarity'] = df['embedding'].apply(lambda x: analysis.calculate_focus_score(x, query_embedding))
        results = df.sort_values(by='similarity', ascending=False).head(5)
        
        for index, row in results.iterrows():
            st.subheader(f"{row['timestamp']} - Similarity: {row['similarity']:.2f}")
            st.write(f"App: {row['active_app_name']} | Window: {row['active_window_title']}")
            if row['screenshot_path'] and os.path.exists(row['screenshot_path']):
                st.image(row['screenshot_path'], width=300)
            st.write("---")
