import tensorflow as tf # <--- ADD THIS LINE
import os
import shutil

# --- Configuration ---
MODEL_FILENAME = 'cnn_model_final.keras'
TARGET_FOLDER = 'model'
PROJECT_ROOT = os.getcwd() 

# --- Execution ---
TARGET_PATH = os.path.join(PROJECT_ROOT, TARGET_FOLDER, MODEL_FILENAME)

# --- (The file move logic is now complete and will likely be skipped) ---
if os.path.exists(TARGET_PATH):
    print(f"✅ Model file already in the correct location: {TARGET_PATH}")
else:
    SOURCE_PATH = os.path.join(PROJECT_ROOT, MODEL_FILENAME) 
    if os.path.exists(SOURCE_PATH):
        os.makedirs(os.path.join(PROJECT_ROOT, TARGET_FOLDER), exist_ok=True)
        try:
            shutil.move(SOURCE_PATH, TARGET_PATH)
            print(f"✅ Successfully moved '{MODEL_FILENAME}' from '{SOURCE_PATH}' to '{TARGET_PATH}'")
        except Exception as e:
            print(f"❌ Error moving file: {e}")
    else:
        print(f"❌ Error: Could not find '{MODEL_FILENAME}' at the assumed source: {SOURCE_PATH}")
        print("Please check where your model file is saved and update the SOURCE_PATH if necessary.")

# --- Load the model (Now with 'tf' defined) ---
try:
    # Use the target path which is now guaranteed to be correct after the move
    model = tf.keras.models.load_model(TARGET_PATH) 
    print("\n✅ Model loaded successfully!")
    
    # Optional: Display model summary to confirm
    model.summary()
    
except Exception as e:
    print(f"\n❌ Final load attempt failed: {e}")
    print("Ensure the file is a valid Keras model and not corrupted.")