# KLYPO — RunPod Serverless
# CUDA 12.8 + PyTorch 2.7 → soporta GPUs Blackwell sm_120 (RTX PRO 6000, B200, B100)
# También compatible con Hopper sm_90, Ampere sm_86/sm_80, Ada sm_89
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── Sistema: Python 3.11 + herramientas de video/audio/fuentes ───────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    # Video
    ffmpeg \
    # Audio (scipy, pyannote)
    libsndfile1 \
    # OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    # Subtítulos: libass busca fuentes vía fontconfig
    libfontconfig1 \
    fontconfig \
    # Utilidades
    git \
    curl \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch 2.7 con CUDA 12.8 — DEBE IR ANTES que requirements.txt ───────────
# Razón: pyannote.audio tiene torch como dependencia; si pip lo resuelve solo
# puede bajar una versión CPU o incompatible. Al instalarlo primero, pip lo
# respeta y no lo sobreescribe al instalar pyannote ni ningún otro paquete.
#
# cu128 incluye kernels compilados para:
#   sm_120 (Blackwell)  ← la que faltaba con cu118
#   sm_90  (Hopper)
#   sm_89  (Ada Lovelace)
#   sm_86/sm_80 (Ampere)
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Resto de dependencias Python ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código de la aplicación ───────────────────────────────────────────────────
COPY modulos_virales/ ./modulos_virales/
COPY fonts/           ./fonts/
COPY handler.py       .

# Variables de entorno — configurar en el panel de RunPod, NO en el contenedor:
#   GROQ_API_KEY        — Llama (detección de clips) + Whisper (b-roll)
#   ASSEMBLYAI_API_KEY  — transcripción con timestamps de palabras
#   HF_TOKEN            — pyannote.audio (diarización, módulo podcasts)
#   OPENAI_API_KEY      — fallback transcripción podcasts
#   PEXELS_API_KEY      — b-roll automático
#   YOUTUBE_COOKIES     — contenido de llave.txt para descargar videos privados/con edad

# Caché HuggingFace persistente dentro del contenedor
# (en RunPod usar Network Volume montado en /app/.cache para no re-descargar modelos)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# -u desactiva el buffer de stdout → logs visibles en tiempo real en RunPod
CMD ["python", "-u", "handler.py"]
