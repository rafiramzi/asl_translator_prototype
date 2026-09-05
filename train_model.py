"""
train_model.py
Skrip paling sederhana untuk melatih model deteksi gesture ASL (huruf & angka).

Alur:
1. Baca semua gambar dari folder dataset (struktur: dataset/<label>/*.jpeg)
2. Ekstrak 21 titik landmark tangan (x, y, z) dengan MediaPipe -> 63 fitur per gambar
3. Latih RandomForestClassifier di atas fitur tsb
4. Simpan model + daftar label ke model.pkl

Cara pakai:
    python train_model.py --data_dir dataset/asl_dataset
"""

import os
import argparse
import pickle
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Hanya folder dengan nama 1 karakter (0-9, a-z) yang dianggap kelas label.
# Ini untuk menghindari folder duplikat/nested di beberapa dataset Kaggle.
VALID_LABEL_LEN = 1

MODEL_ASSET_PATH = "hand_landmarker.task"
MODEL_ASSET_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model_asset():
    """Download model landmark tangan MediaPipe kalau belum ada di lokal."""
    if not os.path.exists(MODEL_ASSET_PATH):
        print(f"Mengunduh model landmark tangan ke {MODEL_ASSET_PATH} ...")
        urllib.request.urlretrieve(MODEL_ASSET_URL, MODEL_ASSET_PATH)
        print("Selesai mengunduh.")


def create_hand_landmarker():
    ensure_model_asset()
    base_options = mp_python.BaseOptions(
        model_asset_path=MODEL_ASSET_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU,  # hindari crash GPU/Metal delegate di macOS
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image_path, detector):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]  # tangan pertama yang terdeteksi
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords  # 21 titik x 3 = 63 fitur


def build_dataset(data_dir):
    detector = create_hand_landmarker()
    X, y = [], []
    skipped = 0

    labels = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)) and len(d) == VALID_LABEL_LEN])

    print(f"Ditemukan {len(labels)} kelas: {labels}")

    for label in labels:
        folder = os.path.join(data_dir, label)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  Memproses kelas '{label}' ({len(files)} gambar)...")

        for fname in files:
            fpath = os.path.join(folder, fname)
            coords = extract_landmarks(fpath, detector)
            if coords is None:
                skipped += 1
                continue
            X.append(coords)
            y.append(label)

    print(f"Total sampel valid: {len(X)}, dilewati (tangan tidak terdeteksi): {skipped}")
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path folder dataset (berisi subfolder per label)")
    parser.add_argument("--output", type=str, default="model.pkl", help="Path output model")
    args = parser.parse_args()

    print("Mengekstrak landmark dari dataset...")
    X, y = build_dataset(args.data_dir)

    if len(X) == 0:
        raise SystemExit("Tidak ada sampel valid. Cek kembali struktur folder dataset.")

    # Buang kelas yang sampelnya terlalu sedikit untuk di-split (butuh minimal 2 agar stratify jalan)
    labels_unique, counts = np.unique(y, return_counts=True)
    valid_labels = set(labels_unique[counts >= 2])
    mask = np.array([label in valid_labels for label in y])
    if (~mask).sum() > 0:
        dropped = set(y[~mask])
        print(f"Melewati kelas dengan sampel < 2 (tidak cukup untuk split): {dropped}")
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Melatih model RandomForest...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi di data test: {acc * 100:.2f}%")

    with open(args.output, "wb") as f:
        pickle.dump({"model": clf, "labels": sorted(set(y))}, f)

    print(f"Model disimpan ke: {args.output}")


if __name__ == "__main__":
    main()