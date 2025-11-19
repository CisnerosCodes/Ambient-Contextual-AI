# Ambient Contextual AI

This project is a personal productivity assistant that runs locally on your machine. The AI "ambiently" observes your on-screen activity to build a rich contextual understanding of your work patterns. It then generates a daily report that provides actionable insights into your productivity, focus, and routines, helping you identify areas for improvement.

## Features

*   **Background Activity Sensing:** A script runs in the background, capturing screenshots, active window titles, and application names.
*   **Rule-Based Classification:** Activities are automatically categorized into tags like "Coding", "Email", and "Entertainment".
*   **Computer Vision Anomaly Detection:** A convolutional autoencoder is trained on your screenshots to learn your "normal" visual routines. It can then detect and flag anomalous activities based on visual deviation.
*   **Interactive Dashboard:** A web-based dashboard built with Streamlit visualizes your daily productivity, including a breakdown of time spent per activity and a timeline of your day.
*   **Anomaly Report:** The dashboard displays the top 5 most anomalous screenshots from the day, allowing you to review moments that didn't fit your usual patterns.

## Tech Stack

*   **Language:** Python
*   **Data Collection:** `mss`, `py-get-active-window`
*   **Database:** SQLite
*   **Machine Learning:** TensorFlow/Keras, Scikit-learn
*   **Dashboard:** Streamlit, Plotly
*   **Image Processing:** OpenCV, Pillow
*   **OCR:** Tesseract (optional, as the core anomaly detection is visual)

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/CisnerosCodes/Ambient-Contextual-AI.git
cd Ambient-Contextual-AI
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR (Optional)
For Optical Character Recognition (OCR) from screenshots, you need to install Google's Tesseract engine.
*   **Windows:** Download and run the installer from [here](https://github.com/UB-Mannheim/tesseract/wiki).
*   After installation, you **must** update the path to the Tesseract executable inside `sensor.py`. The default is `C:\Program Files\Tesseract-OCR\tesseract.exe`.

### 4. Run the Application
The application runs in three main stages:

**Stage 1: Collect Data**
Run the sensor to start collecting data. Let it run in the background while you use your computer. The more data you collect, the better the model will be.
```bash
python sensor.py
```
When you have enough data, stop the script with `Ctrl+C`.

**Stage 2: Train the Anomaly Detection Model**
Run the trainer script. This will process all your collected screenshots and train the autoencoder model. This may take a long time.
```bash
python autoencoder_trainer.py
```

**Stage 3: View the Dashboard**
Once the model is trained, you can view the dashboard.
```bash
streamlit run dashboard.py
```
This will open the dashboard in your web browser.

## File Structure

*   `database.py`: Sets up the initial SQLite database.
*   `sensor.py`: The background script for collecting screenshots and metadata.
*   `classifier.py`: A script for rule-based activity classification (functionality is now integrated into the dashboard).
*   `autoencoder_trainer.py`: The script to train the computer vision anomaly detection model.
*   `dashboard.py`: The main Streamlit application for visualizing data and anomalies.
*   `activity.db`: The SQLite database where all data is stored.
*   `autoencoder.h5`: The trained anomaly detection model.
*   `screenshots/`: The directory where all captured screenshots are saved.
