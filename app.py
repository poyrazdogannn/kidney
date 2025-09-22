# ==========================
# app.py – Kidney Stone Detection (Taş Var / Taş Yok)
# ==========================

import os
import numpy as np
import cv2
import pydicom
from PIL import Image
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --------------------------
# TensorFlow loglarını azalt (info/warning gizle)
# --------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --------------------------
# Flask Uygulaması
# --------------------------
app = Flask(__name__)

# --------------------------
# Modeli yükle (.h5 formatı)
# --------------------------
MODEL_PATH = "smallcnn_224_best.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model başarıyla yüklendi:", MODEL_PATH)
except Exception as e:
    print("❌ Model yüklenemedi:", e)

# Sınıf isimleri (eğitim sırasında kullandığınız sıraya göre)
class_names = ["Taş Yok", "Taş Var"]
IMG_SIZE = (224, 224)


# --------------------------
# Yardımcı Fonksiyonlar
# --------------------------
def preprocess_image(img: Image.Image):
    """Normal resim (jpg/png) -> modele uygun tensör"""
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)


def preprocess_dicom(file):
    """DICOM -> RGB numpy (normalize) -> modele uygun tensör"""
    ds = pydicom.dcmread(file)
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)  # normalize
    arr = cv2.resize(arr, IMG_SIZE)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return np.expand_dims(rgb, axis=0)


def predict(img_tensor):
    preds = model.predict(img_tensor, verbose=0)
    pred_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return class_names[pred_idx], confidence


# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result, conf = None, None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename.lower()
            if filename.endswith(".dcm"):
                img_tensor = preprocess_dicom(file)
            else:
                img = Image.open(file.stream)
                img_tensor = preprocess_image(img)

            result, conf = predict(img_tensor)

    return render_template("index.html", result=result, confidence=conf)


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
