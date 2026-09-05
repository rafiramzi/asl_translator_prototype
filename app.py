"""
app.py
Backend Flask paling sederhana untuk aplikasi deteksi gesture -> text -> speech.

Alur:
1. Frontend kirim frame webcam (base64) ke /predict
2. Backend ekstrak landmark tangan (MediaPipe) & prediksi huruf (model.pkl)
3. Setelah user kumpulkan beberapa huruf, frontend kirim ke /generate_sentence
4. Backend minta Ollama (model gpt-oss:20b cloud) merangkai huruf jadi kalimat rapi
5. Frontend ucapkan hasilnya pakai Web Speech API browser (gratis, tanpa endpoint TTS tambahan)

Menjalankan Ollama cloud model:
    ollama pull gpt-oss:20b-cloud
    ollama run gpt-oss:20b-cloud   (atau cukup pastikan `ollama serve` aktif)
"""

import base64
import os
import pickle
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Load model hasil training ---
with open("model.pkl", "rb") as f:
    saved = pickle.load(f)
    CLF = saved["model"]
    LABELS = saved["labels"]

# --- Setup MediaPipe HandLandmarker (Tasks API) ---
MODEL_ASSET_PATH = "hand_landmarker.task"
MODEL_ASSET_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_ASSET_PATH):
    print(f"Mengunduh model landmark tangan ke {MODEL_ASSET_PATH} ...")
    urllib.request.urlretrieve(MODEL_ASSET_URL, MODEL_ASSET_PATH)

base_options = mp_python.BaseOptions(
    model_asset_path=MODEL_ASSET_PATH,
    delegate=mp_python.BaseOptions.Delegate.CPU,  # hindari crash GPU/Metal delegate di macOS
)
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    running_mode=mp_vision.RunningMode.IMAGE,
)
hands_detector = mp_vision.HandLandmarker.create_from_options(hand_options)

# --- Konfigurasi Ollama ---
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gpt-oss:20b-cloud"  # sesuaikan dengan nama model cloud di Ollama kamu


def decode_base64_image(data_url):
    """Ubah base64 dataURL dari browser jadi array gambar OpenCV (BGR)."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image = decode_base64_image(data["image"])
    if image is None:
        return jsonify({"letter": None, "confidence": 0})

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = hands_detector.detect(mp_image)

    if not result.hand_landmarks:
        return jsonify({"letter": None, "confidence": 0})

    landmarks = result.hand_landmarks[0]
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])

    coords = np.array(coords).reshape(1, -1)
    proba = CLF.predict_proba(coords)[0]
    pred_idx = np.argmax(proba)
    letter = CLF.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    return jsonify({"letter": letter, "confidence": round(confidence, 2)})


@app.route("/generate_sentence", methods=["POST"])
def generate_sentence():
    data = request.get_json()
    letters = data.get("letters", "")

    if not letters.strip():
        return jsonify({"sentence": ""})

    prompt = (
        "Kamu membantu merapikan hasil deteksi bahasa isyarat huruf-per-huruf menjadi "
        "kata atau kalimat yang masuk akal dalam Bahasa Indonesia. "
        f"Huruf-huruf yang terdeteksi (mungkin ada kesalahan kecil): \"{letters}\". "
        "Jawab HANYA dengan hasil kata/kalimat yang sudah dirapikan, tanpa penjelasan tambahan."
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        sentence = result["message"]["content"].strip()
    except Exception as e:
        # Kalau Ollama tidak aktif/error, fallback ke huruf mentah saja
        sentence = letters
        print(f"[WARNING] Gagal menghubungi Ollama: {e}")

    return jsonify({"sentence": sentence})


if __name__ == "__main__":
    app.run(debug=True, port=5000)