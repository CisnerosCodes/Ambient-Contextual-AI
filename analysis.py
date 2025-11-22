import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
import ollama

DB_FILE = "activity.db"
ANCHOR_FILE = "anchor.json"

def get_db_connection():
    return sqlite3.connect(DB_FILE)

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def calculate_focus_score(current_embedding, anchor_embedding):
    """
    Calculate focus score (0-1) based on similarity to anchor state.
    """
    if current_embedding is None or anchor_embedding is None:
        return 0.0
    
    score = cosine_similarity(current_embedding, anchor_embedding)
    # Ensure score is between 0 and 1 (cosine sim is -1 to 1)
    return max(0.0, float(score))

def get_anchor_embedding():
    """Retrieve the anchor embedding from file."""
    try:
        with open(ANCHOR_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def set_anchor_embedding(embedding):
    """Save the anchor embedding to file."""
    with open(ANCHOR_FILE, "w") as f:
        json.dump(embedding, f)

def generate_hourly_summary(model_name='llama3'):
    """
    Fetch OCR text from the last hour and generate a summary using Ollama.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    one_hour_ago = datetime.now() - timedelta(hours=1)
    cursor.execute("SELECT ocr_text FROM logs WHERE timestamp > ?", (one_hour_ago,))
    rows = cursor.fetchall()
    
    if not rows:
        conn.close()
        return "No activity recorded in the last hour."
    
    # Combine text
    combined_text = "\n".join([row[0] for row in rows if row[0]])
    
    # Truncate if too long
    if len(combined_text) > 10000:
        combined_text = combined_text[:10000] + "..."
        
    prompt = f"Based on the following screen text captured over the last hour, summarize what the user was working on. Be concise and narrative. Focus on the main tasks.\n\n{combined_text}"
    
    try:
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': prompt},
        ])
        summary = response['message']['content']
        
        # Store summary
        cursor.execute("INSERT INTO summaries (summary_text) VALUES (?)", (summary,))
        conn.commit()
        return summary
    except Exception as e:
        print(f"DEBUG: Ollama Error: {e}") # Print to terminal for debugging
        error_msg = str(e)
        if "connection refused" in error_msg.lower():
            return "Error: Ollama is not running. Please run 'ollama serve' in a terminal."
        if "model" in error_msg.lower() and "not found" in error_msg.lower():
            return f"Error: Model '{model_name}' not found. Please run 'ollama pull {model_name}' in a terminal."
        return f"Error generating summary: {error_msg}"
    finally:
        conn.close()

if __name__ == "__main__":
    print("Analysis module ready.")
