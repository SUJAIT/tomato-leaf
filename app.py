import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import json
import gdown
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "tomato_leaf_cnn.keras")
CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://drive.google.com/uc?id=1CksOGheGylHFH66PhybBqhYmf0J8d_09"
)

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("⬇ Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

model = None
def get_model():
    global model
    if model is None:
        print("📦 Loading model into memory...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

def predict_image(image_path):
    mdl = get_model()
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = mdl.predict(img)
    class_id = int(np.argmax(preds))
    confidence = float(preds[0][class_id])

    return class_names[class_id], round(confidence * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            image_path = os.path.join("static", file.filename)
            file.save(image_path)
            result, confidence = predict_image(image_path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
