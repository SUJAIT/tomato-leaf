# import os
# import json
# import gdown
# import tensorflow as tf
# import numpy as np
# from flask import Flask, render_template, request
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# app = Flask(__name__)

# # =========================
# # Model download from Drive
# # =========================
# MODEL_DIR = "model"
# MODEL_PATH = "model/tomato_leaf_cnn.keras"
# MODEL_URL = os.environ.get(
#     "MODEL_URL",
#     "https://drive.google.com/uc?id=1CksOGheGylHFH66PhybBqhYmf0J8d_09"
# )

# os.makedirs(MODEL_DIR, exist_ok=True)

# if not os.path.exists(MODEL_PATH):
#     print("⬇ Downloading model from Google Drive...")
#     gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# # Load model
# model = tf.keras.models.load_model(MODEL_PATH)

# # Load class
# with open("model/class_names.json", "r") as f:
#     class_names = json.load(f)

# IMG_SIZE = (224, 224)

# def predict_image(image_path):
#     img = load_img(image_path, target_size=IMG_SIZE)
#     img = img_to_array(img) / 255.0
#     img = np.expand_dims(img, axis=0)

#     preds = model.predict(img)
#     class_id = np.argmax(preds)
#     confidence = float(preds[0][class_id])

#     return class_names[class_id], round(confidence * 100, 2)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     result = None
#     confidence = None
#     image_path = None

#     if request.method == "POST":
#         file = request.files["image"]
#         image_path = os.path.join("static", file.filename)
#         file.save(image_path)

#         result, confidence = predict_image(image_path)

#     return render_template(
#         "index.html",
#         result=result,
#         confidence=confidence,
#         image_path=image_path
#     )

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

# # if __name__ == "__main__":
# #     app.run(debug=True)


import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model/tomato_leaf_cnn.keras")

def predict(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = img.reshape(1,224,224,3)
    pred = model.predict(img)
    return str(np.argmax(pred))

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Tomato Leaf Disease Detection"
).launch()
