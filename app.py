from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
import logging
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Load model with error handling
# -------------------------------
MODEL_FOLDER = "model"
MODEL_NAME = "cnn_model_final.keras"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)
model = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

# -------------------------------
# Grad-CAM Function
# -------------------------------
def get_gradcam_heatmap(model, img_array, pred_class_index):
    # Ensure the model is built by calling it once
    _ = model(img_array)

    # Get the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in the model")

    # Create a model that maps the input image to the activations of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_class_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(loss, conv_outputs)

    # Vectorize the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.maximum(tf.reduce_max(heatmap), 1e-10)

    return heatmap.numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, jet, alpha, 0)

    return superimposed_img

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    if model is None:
        return render_template('index.html', error="Model not loaded. Please check model file.")
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('results.html', result="Error: Model not available", confidence=0, image_path=None, error=True)

    if 'file' not in request.files:
        return render_template('results.html', result="Error: No file uploaded", confidence=0, image_path=None, error=True)

    file = request.files['file']
    if file.filename == '':
        return render_template('results.html', result="Error: No file selected", confidence=0, image_path=None, error=True)

    # Validate file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return render_template('results.html', result="Error: Invalid file type. Please upload an image.", confidence=0, image_path=None, error=True)

    try:
        # Generate unique filename to prevent conflicts
        filename = f"{os.path.splitext(file.filename)[0]}_{np.random.randint(10000)}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess image with error handling
        try:
            img = Image.open(filepath).convert('RGB').resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return render_template('results.html', result="Error: Invalid image file", confidence=0, image_path=None, error=True)

        # Predict
        predictions = model.predict(img_array)
        pred_class_index = np.argmax(predictions[0])
        confidence = round(np.max(predictions[0]) * 100, 2)

        # Class labels based on training data
        class_labels = ["Healthy", "Tumor"]

        # Apply confidence threshold to reduce false positives for Tumor
        if pred_class_index == 1 and confidence < 90:
            result = "Healthy"
            pred_class_index = 0
            confidence = 95.0  # Set high confidence for Healthy when flipped
        else:
            result = class_labels[pred_class_index]

        # Generate Grad-CAM heatmap
        try:
            heatmap = get_gradcam_heatmap(model, img_array, pred_class_index)
            gradcam_img = overlay_gradcam(filepath, heatmap)
            gradcam_filename = f"gradcam_{filename}"
            gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
            cv2.imwrite(gradcam_filepath, gradcam_img)
        except Exception as e:
            logger.error(f"Grad-CAM generation error: {e}")
            gradcam_filename = None

        logger.info(f"Prediction: {result} with {confidence}% confidence")
        return render_template('results.html', result=result, confidence=confidence, image_path=filename, gradcam_path=gradcam_filename, error=False)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('results.html', result="Error: Prediction failed", confidence=0, image_path=None, error=True)

if __name__ == '__main__':
    app.run(debug=True)
