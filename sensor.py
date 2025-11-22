import sqlite3
import time
import os
import json
from datetime import datetime
import mss
import pygetwindow as gw
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer

# --- Configuration ---
DB_FILE = "activity.db"
SCREENSHOTS_DIR = "screenshots"
CAPTURE_INTERVAL = 10  # seconds

# --- Tesseract OCR Configuration ---
# IMPORTANT: You need to install Tesseract OCR on your system.
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Default Windows installation path:
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # Fallback: Assume it's in the system PATH
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

def create_database_connection(db_file):
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn

def insert_log(conn, data):
    """
    Log a new activity record to the database.
    :param conn: Connection object
    :param data: A tuple containing (active_app_name, active_window_title, screenshot_path, ocr_text, embedding_json)
    """
    sql = ''' INSERT INTO logs(active_app_name, active_window_title, screenshot_path, ocr_text, embedding_json)
              VALUES(?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, data)
    conn.commit()
    return cur.lastrowid

def capture_screenshot(sct, screenshots_dir):
    """Capture the screen and save it to a file."""
    # Create daily folder
    date_folder = os.path.join(screenshots_dir, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(date_folder, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%H-%M-%S")
    filename = f"{timestamp}.png"
    filepath = os.path.join(date_folder, filename)

    # Capture and save
    sct.shot(output=filepath)
    return filepath

def get_active_window_info():
    """Get the active application name and window title."""
    try:
        active_window = gw.getActiveWindow()
        if active_window:
            # On Windows, active_window.title gives the title.
            # To get the app name, we can try to parse the exe from the window handle if needed,
            # but for simplicity, we'll use what pygetwindow provides.
            # This part might need adjustment depending on the OS and desired granularity.
            app_name = active_window.title # Placeholder, might need a better way to get app name
            window_title = active_window.title
            return app_name, window_title
    except Exception as e:
        print(f"Could not get active window: {e}")
    return "Unknown", "Unknown"


def perform_ocr(image_path):
    """Perform OCR on an image and return the extracted text."""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""

def main():
    """Main loop to capture and log activity."""
    # Ensure screenshots directory exists
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    # Initialize Embedding Model
    print("Loading Embedding Model (CLIP)...")
    model = SentenceTransformer('clip-ViT-B-32')
    print("Model loaded.")

    db_conn = create_database_connection(DB_FILE)
    if not db_conn:
        print("Error: Could not connect to the database. Exiting.")
        return

    with mss.mss() as sct:
        while True:
            print(f"--- {datetime.now()}: Capturing activity ---")

            # 1. Capture Screenshot
            screenshot_path = capture_screenshot(sct, SCREENSHOTS_DIR)
            print(f"Screenshot saved to: {screenshot_path}")

            # 2. Get Active Window Info
            app_name, window_title = get_active_window_info()
            print(f"Active Window: {app_name} - {window_title}")

            # 3. Perform OCR
            ocr_text = perform_ocr(screenshot_path)
            print(f"OCR Text (first 50 chars): {ocr_text[:50].strip()}...")

            # 4. Generate Embedding
            image = Image.open(screenshot_path)
            embedding = model.encode(image)
            embedding_json = json.dumps(embedding.tolist())

            # 5. Save to Database
            log_data = (app_name, window_title, screenshot_path, ocr_text, embedding_json)
            insert_log(db_conn, log_data)
            print("Activity logged to database.")

            # 5. Wait for the next interval
            time.sleep(CAPTURE_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\nMissing dependency: {e.name}")
        print("Please install the required libraries by running:")
        print("pip install mss pygetwindow pytesseract Pillow")
    except FileNotFoundError:
        print("\nError: Tesseract OCR not found.")
        print("Please install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki")
        print("And update the 'pytesseract.pytesseract.tesseract_cmd' path in sensor.py.")
    except KeyboardInterrupt:
        print("\nSensor stopped by user.")
