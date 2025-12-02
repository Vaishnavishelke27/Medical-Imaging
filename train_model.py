import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 

# --- PATH CONFIGURATION ---
DATASET_PATH = os.path.normpath(r"C:\Users\VAISHNAVI\Downloads\archive (6)\Dataset\Brain Tumor CT scan Images")
MODEL_SAVE_DIR = 'model' # The folder where the model must go

# Ensure the directory exists
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
    print(f"Created directory: {MODEL_SAVE_DIR}/")

# --- PARAMETERS ---
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
MAX_EPOCHS = 50 # Increased epochs, using EarlyStopping for control

# --- DATA PREPROCESSING ---
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

NUM_CLASSES = train_gen.num_classes

# --- CNN MODEL DEFINITION ---
model = Sequential([
    # Input Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    
    # Mid Block 1
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Mid Block 2
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Final Feature Extraction Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    
    # Fully Connected Layers
    Dense(256, activation='relu'),
    Dropout(0.5),
    
    # Output Layer
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- CALLBACKS (WITH CORRECTED SAVE PATHS) ---

# Corrected save path for best model
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, 'cnn_model_best.keras')

checkpoint = ModelCheckpoint(
    CHECKPOINT_PATH, 
    monitor='val_accuracy', 
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

# --- MODEL TRAINING ---
print("\n--- Starting Model Training ---")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=MAX_EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

# Corrected save path for final model
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'cnn_model_final.keras')
model.save(FINAL_MODEL_PATH)

print(f"\n✅ Training complete. Model saved to '{FINAL_MODEL_PATH}'")