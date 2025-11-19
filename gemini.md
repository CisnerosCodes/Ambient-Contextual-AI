Project Plan: Ambient Contextual AI (ACA)

1.  Project Objective
    To build a personal productivity assistant that runs locally on my machine. The AI will "ambiently" observe my on-screen activity to build a rich contextual understanding of my work patterns. The final goal is to generate a daily report that provides actionable insights into my productivity, focus, and routines, helping me identify areas for improvement.

2.  Core Tech Stack
    *   **Language:** Python
    *   **Data Collection:** mss (for fast screenshots), py-get-active-window (for app context), pytesseract (for OCR).
    *   **Database:** SQLite (simple, local, and file-based).
    *   **ML/CV Model:** TensorFlow/Keras.
    *   **Dashboard:** Streamlit (fast to build a local web app).

3.  Development Phases

    ---
    **Phase 1: The Sensor (Data Collection Layer)** - *Completed*
    *Goal: Create a robust background script (sensor.py) that captures all necessary data and stores it in a local SQLite database.*

    *   **Task 1.1: Setup Database** - *Completed*
        *   Create `database.py` and define `activity.db` with a `logs` table.
    *   **Task 1.2: Build sensor.py** - *Completed*
        *   Create a script to capture screenshots, window info, and OCR text, saving it all to the database every 10 seconds.

    ---
    **Phase 2: The Brain (Machine Learning Layer)**
    *Goal: Analyze the collected data to classify activities and identify anomalies using computer vision.*

    *   **Task 2.1: Simple Activity Classification** - *Completed*
        *   Create `classifier.py` to read logs and apply rule-based classification (e.g., "Coding", "Email").
        *   Save results to `classified_activity.csv`.

    *   **Task 2.2: CV Anomaly Detector Model (Autoencoder)** - *In Progress*
        *   Create `autoencoder_trainer.py`.
        *   **Load & Preprocess:** Load all screenshots, resize them to a uniform size (e.g., 128x128), and convert to grayscale.
        *   **Build Model:** Use TensorFlow/Keras to build a convolutional autoencoder.
        *   **Train Model:** Train the autoencoder on the preprocessed screenshots to learn a representation of "normal" activity.
        *   **Save Model:** Save the trained model to `autoencoder.h5`.

    ---
    **Phase 3: The Dashboard (Visualization Layer)**
    *Goal: Create an interactive dashboard to display productivity insights and anomalies.*

    *   **Task 3.1: Dashboard UI** - *Completed*
        *   Create `dashboard.py` using Streamlit.
        *   Add a date selector to query and display data from `activity.db` for the selected day.

    *   **Task 3.2: Productivity Visualization** - *Completed*
        *   **Pie Chart:** Add a chart showing the breakdown of time spent in each activity category.
        *   **Timeline:** Add a bar chart showing the dominant activity for every 10-minute block of the day.

    *   **Task 3.3: Anomaly Report** - *Up Next*
        *   Load the trained `autoencoder.h5` model.
        *   For the selected day, run all screenshots through the model and calculate their reconstruction error (MSE).
        *   **Display:** Show the "Top 5 Anomalies" for the day (the 5 screenshots with the highest error), along with their timestamp and a warning message.
