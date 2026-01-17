import os
import json
import gdown
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# =========================
# Config (IMPORTANT)
# =========================
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB upload limit

# =========================
# Model download from Drive
# =========================
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "tomato_leaf_cnn.keras")
CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://drive.google.com/uc?id=1CksOGheGylHFH66PhybBqhYmf0J8d_09"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("⬇ Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# =========================
# Load model (SAFE)
# =========================
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# =========================
# Load class names
# =========================
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

# =========================
# Prediction function
# =========================
def predict_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = int(np.argmax(preds))
    confidence = float(preds[0][class_id])

    return class_names[class_id], round(confidence * 100, 2)

# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html")

        image_path = os.path.join("static", file.filename)
        file.save(image_path)

        result, confidence = predict_image(image_path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )

# =========================
# Main
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
