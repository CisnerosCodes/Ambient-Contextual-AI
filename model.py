import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import joblib

# --- Configuration ---
INPUT_CSV = "classified_activity.csv"
OUTPUT_CSV = "anomaly_results.csv"
MODEL_FILE = "anomaly_model.joblib"

def load_and_prepare_data(filepath):
    """Load the classified data and engineer features for the model."""
    df = pd.read_csv(filepath)
    
    # Convert timestamp to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature Engineering
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek # Monday=0, Sunday=6
    
    # One-Hot Encode the 'category' column
    encoder = OneHotEncoder(handle_unknown='ignore')
    category_encoded = encoder.fit_transform(df[['category']]).toarray()
    encoded_df = pd.DataFrame(category_encoded, columns=encoder.get_feature_names_out(['category']))
    
    # Combine original numerical features with new encoded features
    features_df = df[['hour_of_day', 'day_of_week']]
    final_df = pd.concat([features_df, encoded_df], axis=1)
    
    return df, final_df

def train_anomaly_model(features):
    """Train an Isolation Forest model."""
    # The 'contamination' parameter is the expected proportion of anomalies.
    # This is a key parameter to tune. 'auto' is a good starting point.
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(features)
    return model

def main():
    """Main function to train the model and identify anomalies."""
    print("Starting anomaly detection model training...")

    # 1. Load and prepare data
    try:
        original_df, features_df = load_and_prepare_data(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV}'.")
        print("Please run the classifier script first (classifier.py).")
        return
        
    if features_df.empty:
        print("No data available for training. Exiting.")
        return

    print(f"Loaded and prepared {len(original_df)} records.")
    print("Features for model:", features_df.columns.tolist())

    # 2. Train the model
    model = train_anomaly_model(features_df)
    print("Anomaly detection model trained successfully.")

    # 3. Predict anomalies
    # The model returns -1 for anomalies and 1 for inliers.
    predictions = model.predict(features_df)
    original_df['is_anomaly'] = [1 if p == -1 else 0 for p in predictions]
    
    print("Anomaly prediction complete.")
    
    # 4. Save the model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # 5. Save the results
    original_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results with anomaly flags saved to {OUTPUT_CSV}")

    # Display anomalies found
    anomalies = original_df[original_df['is_anomaly'] == 1]
    if not anomalies.empty:
        print("\n--- Anomalies Detected ---")
        print(anomalies[['timestamp', 'active_window_title', 'category']])
    else:
        print("\nNo anomalies were detected in this dataset.")


if __name__ == "__main__":
    main()
