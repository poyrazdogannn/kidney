from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import pydicom, cv2, os

app = Flask(__name__)

# Modeli yükle
model = tf.keras.models.load_model("smallcnn_224_best.keras", compile=False)
class_names = ["Taş Yok", "Taş Var"]
IMG_SIZE = (224, 224)

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
    idx = np.argmax(preds, axis=1)[0]
    conf = float(np.max(preds))
    return class_names[idx], conf

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return render_template("index.html", result="Dosya yüklenmedi.", conf=None)
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix == ".dcm":
            img_tensor = preprocess_dicom(file)
        else:
            img = Image.open(file.stream)
            img_tensor = preprocess_image(img)
        result, conf = predict(img_tensor)
        return render_template("index.html", result=result, conf=conf)

    return render_template("index.html", result=None, conf=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
