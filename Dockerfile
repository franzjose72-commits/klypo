# KLYPO - RunPod Serverless
# Base: RunPod PyTorch 2.7.1 + CUDA 12.8.1 + Ubuntu 22.04
# PyTorch ya viene preinstalado con sm_120 (Blackwell) - no hay que instalarlo desde cero
FROM runpod/pytorch:1.0.7-cu1281-torch271-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive

# Solo lo que falta en la imagen base de RunPod
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    fontconfig \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Deno: runtime JS que yt-dlp necesita para resolver el challenge "n" de YouTube
RUN curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
    -o /tmp/deno.zip \
    && unzip /tmp/deno.zip -d /usr/local/bin/ \
    && chmod +x /usr/local/bin/deno \
    && rm /tmp/deno.zip \
    && deno --version

WORKDIR /app

# Dependencias Python
# torch 2.7.1 ya esta en la imagen base - pip lo detecta instalado y no lo vuelve a bajar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de la aplicacion
COPY modulos_virales/ ./modulos_virales/
COPY fonts/           ./fonts/
COPY handler.py       .

# Cache HuggingFace (usar Network Volume en /app/.cache en RunPod para no re-descargar modelos)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

CMD ["python", "-u", "handler.py"]
