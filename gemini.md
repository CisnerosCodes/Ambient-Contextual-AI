Project Plan: Ambient Contextual AI (Genius Edition)

1. Project Objective

To build a "Zero-Training" personal productivity assistant. Instead of complex model training, we use Semantic Embeddings (CLIP) to mathematically measure focus and Local LLMs to generate human-readable context summaries of daily activity.

2. Core Tech Stack

Language: Python

Vision/Embedding: sentence-transformers (using the CLIP model clip-ViT-B-32).

Context/Summary: ollama (running llama3 or gemma:2b locally).

Data Collection: mss, py-get-active-window, pytesseract.

Database: SQLite (storing Vectors and Logs).

Dashboard: Streamlit.

3. Development Phases

Phase 1: The Semantic Sensor (Data Collection)

Goal: Capture screen activity and immediately convert it into "meaning" (Vectors) and text.

Task 1.1: Setup Smart Database (database.py)

Table logs:

id, timestamp, app_name, window_title

ocr_text (The raw text content)

embedding_json (The CLIP vector stored as a JSON text string)

screenshot_path (Optional, for review)

Task 1.2: The Embedding Engine (sensor.py)

Initialize SentenceTransformer('clip-ViT-B-32').

Loop (10s):

Capture Screenshot.

Generate Embedding: vector = model.encode(image).

OCR: Extract text.

Store Vector, Text, and App Name in DB.

Phase 2: The Context Engine (Analysis)

Goal: Turn raw vectors and text into "Focus Scores" and "Summaries" without training.

Task 2.1: Focus Calculator (analysis.py)

The Golden Anchor: Create a script to capture one screenshot of your "Ideal Work State" and save its vector as reference_vector.

Calculate Focus: For every log, calculate Cosine Similarity between current_vector and reference_vector.

Score (0-1): 1.0 is perfect focus, 0.0 is complete context switch.

Task 2.2: The Narrator (LLM Summary)

Requirement: Install Ollama (https://ollama.com/).

Create a function generate_hourly_summary():

Fetch all ocr_text for the last hour.

Send to Ollama: "Based on this screen text, what was the user doing?"

Store the result in a new table summaries.

Phase 3: The Intelligence Dashboard

Goal: A UI that tells you the story of your day.

Task 3.1: Streamlit UI

The Focus Wave: A line chart of your "Focus Score" over time. High plateaus = Deep Work. Valleys = Distractions.

The Narrative: Display the hourly LLM summaries (e.g., "9 AM - 10 AM: Working on Python Database scripts").

Semantic Search: A text box where you can type "Reading News" and it searches your database vectors to find when you were doing that.