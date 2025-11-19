import sqlite3
import pandas as pd

DB_FILE = "activity.db"
OUTPUT_CSV = "classified_activity.csv"

def load_data_from_db(db_file):
    """Loads all records from the logs table into a pandas DataFrame."""
    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query("SELECT * FROM logs", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return pd.DataFrame()

def classify_activity(row):
    """
    Classifies a single activity record based on predefined rules.
    """
    app_name = row['active_app_name'].lower()
    window_title = row['active_window_title'].lower()
    ocr_text = str(row['ocr_text']).lower() if pd.notna(row['ocr_text']) else ""

    # Rule-based classification
    if any(keyword in app_name for keyword in ["code.exe", "visual studio code", "sublime_text"]):
        return "Coding"
    if any(keyword in window_title for keyword in ["outlook", "gmail"]):
        return "Email"
    if any(keyword in window_title for keyword in ["youtube", "netflix"]):
        return "Entertainment"
    if any(keyword in ocr_text for keyword in ["error", "debug", "traceback"]):
        return "Debugging"
    
    # Default category
    return "General"

def main():
    """Main function to load, classify, and save activity data."""
    print("Starting activity classification...")

    # 1. Load data
    df = load_data_from_db(DB_FILE)
    if df.empty:
        print("No data found in the database. Exiting.")
        return

    print(f"Loaded {len(df)} records from the database.")

    # 2. Classify data
    df['category'] = df.apply(classify_activity, axis=1)
    print("Activity classification complete.")
    print("\nClassification Summary:")
    print(df['category'].value_counts())

    # 3. Save classified data
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nClassified data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
