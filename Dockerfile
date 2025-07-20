# ✅ Base image with CUDA, cuDNN, libdevice, and dev tools
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# === 🧰 System tools and Python ===
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git curl wget unzip nano build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.10 /usr/bin/python && ln -sf pip3 /usr/bin/pip

# === 📦 Python dependencies ===
COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# === 🗂️ Preload NLTK resources ===
RUN python -m nltk.downloader wordnet omw-1.4

# === 🧠 App code ===
COPY ./lucenai /app/lucenai
COPY ./scripts/train.py /app/scripts/train.py

# === ⚙️ CUDA compatibility ===
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV TF_CPP_MIN_LOG_LEVEL=2

# === 🚀 Entry point ===
# CMD ["python", "scripts/train.py"]
# Force fine-tuned model building
CMD ["python", "scripts/train.py", "-f"]
# Run distillation:
# CMD ["python", "scripts/train.py", "-d"]
# Force fine-tuned model building and run distillation:
# CMD ["python", "scripts/train.py", "-d", "-f"]