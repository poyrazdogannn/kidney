# ==========================
# app.py – Kidney Stone Detection (Taş Var / Taş Yok)
# ==========================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import pydicom
import cv2
import os

# --------------------------
# GPU devre dışı bırak (Render CPU ortamı)
# --------------------------
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

# --------------------------
# Modeli yükle
# --------------------------
@st.cache_resource
def load_model():
    model_file = None
    if os.path.exists("smallcnn_224_best.h5"):
        model_file = "smallcnn_224_best.h5"
    elif os.path.exists("smallcnn_224_best.keras"):
        model_file = "smallcnn_224_best.keras"

    if model_file is None:
        raise FileNotFoundError("❌ Model dosyası bulunamadı (smallcnn_224_best.h5 veya .keras).")

    st.write(f"📂 Model yükleniyor: `{model_file}`")
    return tf.keras.models.load_model(model_file, compile=False)

try:
    model = load_model()
    MODEL_OK = True
except Exception as e:
    st.error(f"🚨 Model yüklenemedi: {e}")
    MODEL_OK = False

# Sınıf isimleri (eğitim sırasında kullandığınız sıraya göre kontrol edin!)
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

def preprocess_dicom(file):
    ds = pydicom.dcmread(file)
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
# Streamlit UI
# --------------------------
st.title("💎 Böbrek Taşı Tespit Uygulaması")
st.write("Resim veya DICOM yükleyin, model **Taş Var** / **Taş Yok** sonucunu verecek.")

if MODEL_OK:
    uploaded_file = st.file_uploader("Bir dosya yükleyin (jpg/png/dcm)", type=["jpg", "jpeg", "png", "dcm"])

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        if suffix == ".dcm":
            st.info("📂 DICOM dosyası yüklendi")
            img_tensor = preprocess_dicom(uploaded_file)
            result, conf = predict(img_tensor)
            st.subheader(f"🔎 Tahmin: **{result}** (Güven: {conf:.2f})")

        else:
            img = Image.open(uploaded_file)
            st.image(img, caption="Yüklenen Görüntü", use_column_width=True)
            img_tensor = preprocess_image(img)
            result, conf = predict(img_tensor)
            st.subheader(f"🔎 Tahmin: **{result}** (Güven: {conf:.2f})")
else:
    st.warning("⚠️ Model yüklenemediği için tahmin yapılamıyor.")
