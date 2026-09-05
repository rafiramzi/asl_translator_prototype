FROM python:3.11-slim

# Dependency sistem yang dibutuhkan OpenCV & mediapipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    libglvnd0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Untuk training: docker run --rm -v $(pwd):/app sign-app python train_model.py --data_dir ./datasets --output model.pkl
# Untuk app:      docker run --rm -p 5000:5000 -v $(pwd):/app sign-app python app.py
CMD ["python", "app.py"]