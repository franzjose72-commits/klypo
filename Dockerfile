# KLYPO - RunPod Serverless
# Base: RunPod PyTorch 2.7.1 + CUDA 12.8.1 + Ubuntu 22.04
# PyTorch ya viene preinstalado con sm_120 (Blackwell) - no hay que instalarlo desde cero
FROM runpod/pytorch:1.0.7-cu1281-torch271-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive

# Sistema: ffmpeg, OpenCV, fuentes, curl, git, unzip
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
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 LTS - yt-dlp lo necesita para bgutil PO token provider
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version \
    && npm --version

# bgutil server: genera los PO tokens que YouTube exige para SABR/web client
# Se instala en /root/bgutil-ytdlp-pot-provider (ruta por defecto que busca yt-dlp)
RUN git clone --single-branch --branch 1.3.1 \
    https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git \
    /root/bgutil-ytdlp-pot-provider \
    && cd /root/bgutil-ytdlp-pot-provider/server \
    && npm ci \
    && npx tsc \
    && echo "bgutil server listo en /root/bgutil-ytdlp-pot-provider/server/build/"

WORKDIR /app

# Dependencias Python (torch 2.7.1 ya en imagen base - pip lo detecta y lo omite)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de la aplicacion
COPY modulos_virales/ ./modulos_virales/
COPY fonts/           ./fonts/
COPY handler.py       .

# Cache HuggingFace (usar Network Volume en /app/.cache en RunPod para persistir modelos)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

CMD ["python", "-u", "handler.py"]
