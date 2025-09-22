# ==========================
# app.py – Flask + TensorFlow (Taş Var / Taş Yok)
# ==========================

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import pydicom
import cv2
import os

app = Flask(__name__)

# --------------------------
# Modeli yükle
# --------------------------
model_file = None
if os.path.exists("smallcnn_224_best.h5"):
    model_file = "smallcnn_224_best.h5"
elif os.path.exists("smallcnn_224_best.keras"):
    model_file = "smallcnn_224_best.keras"
elif os.path.exists("smallcnn_224_best.H5"):
    model_file = "smallcnn_224_best.H5",
elif os.path.exists("smallcnn_224_best.KERAS"):
    model_file = "smallcnn_224_best.KERAS"
    
    
if model_file is None:
    raise FileNotFoundError("❌ Model dosyası bulunamadı!")

model = tf.keras.models.load_model(model_file, compile=False)

class_names = ["Taş Yok", "Taş Var"]
IMG_SIZE = (224, 224)

# --------------------------
# Yardımcı Fonksiyonlar
# --------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)

def preprocess_dicom(file_path):
    ds = pydicom.dcmread(file_path)
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
    arr = cv2.resize(arr, IMG_SIZE)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return np.expand_dims(rgb, axis=0)

def predict(img_tensor):
    preds = model.predict(img_tensor, verbose=0)
    pred_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return class_names[pred_idx], confidence

# --------------------------
# Flask Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filename = file.filename.lower()
            filepath = os.path.join("uploads", filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(filepath)

            if filename.endswith(".dcm"):
                img_tensor = preprocess_dicom(filepath)
            else:
                img = Image.open(filepath)
                img_tensor = preprocess_image(img)

            result, confidence = predict(img_tensor)

    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
