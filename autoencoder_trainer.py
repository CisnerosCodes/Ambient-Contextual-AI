import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import sys

# --- Configuration ---
SCREENSHOTS_DIR = "screenshots"
MODEL_PATH = "autoencoder.h5"
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

def load_and_preprocess_images(image_dir):
    """Load all screenshots, resize them, and preprocess for the model."""
    print(f"Loading images from: {image_dir}")
    image_paths = glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True)
    
    if not image_paths:
        print("Error: No images found. Please ensure the sensor has run and captured screenshots.", file=sys.stderr)
        return None

    processed_images = []
    for path in image_paths:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            img = img.astype('float32') / 255.0
            processed_images.append(img)
        except Exception as e:
            print(f"Could not process image {path}: {e}")
            
    return np.array(processed_images).reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

def build_autoencoder(input_shape):
    """Build a convolutional autoencoder model."""
    model = models.Sequential()
    
    # Encoder
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    
    # Decoder
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    
    # Output Layer
    model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

def main():
    """Main function to load data, train the model, and save it."""
    print("Starting autoencoder training...")
    
    # 1. Load Data
    train_data = load_and_preprocess_images(SCREENSHOTS_DIR)
    if train_data is None or len(train_data) < 10:
        print("Not enough images to train. Need at least 10. Exiting.", file=sys.stderr)
        return

    print(f"Successfully loaded and processed {len(train_data)} images.")

    # 2. Build Model
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    autoencoder = build_autoencoder(input_shape)

    # 3. Train Model
    print("\n--- Training Model ---")
    # We train the model to reconstruct its own input.
    # A portion of the data is used for validation to monitor for overfitting.
    autoencoder.fit(train_data, train_data,
                    epochs=20,
                    batch_size=16,
                    shuffle=True,
                    validation_split=0.2)
    
    # 4. Save Model
    print("\n--- Saving Model ---")
    autoencoder.save(MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
