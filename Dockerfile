# KLYPO — RunPod Serverless
# Base: imagen oficial RunPod con PyTorch 2.1 + CUDA 11.8 + Python 3.10
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ── Dependencias de sistema ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Video / audio
    ffmpeg \
    libsndfile1 \
    # OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    # Subtítulos / fuentes (libass usa fontconfig para buscar TTFs)
    libfontconfig1 \
    fontconfig \
    # Utilidades
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencias Python ───────────────────────────────────────────────────────
COPY requirements.txt .

# 1. Instalar paquetes del requirements.txt (sin torch — lo especificamos con CUDA abajo)
RUN pip install --no-cache-dir -r requirements.txt

# 2. PyTorch con CUDA 11.8 (reemplaza cualquier versión CPU que haya entrado)
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. RunPod Serverless SDK
RUN pip install --no-cache-dir runpod>=1.5.0

# ── Código de la aplicación ───────────────────────────────────────────────────
COPY modulos_virales/ ./modulos_virales/
COPY fonts/           ./fonts/
COPY handler.py       .

# .env NO se copia — las variables de entorno se configuran en el panel de RunPod:
#   GROQ_API_KEY, HF_TOKEN, OPENAI_API_KEY, ASSEMBLYAI_API_KEY, PEXELS_API_KEY
#
# llave.txt (cookies de YouTube) NO se copia — si tu contenido de YouTube lo requiere,
# monta un volumen de red en RunPod y ajusta la ruta en motor_viral.py:
#   cookies = os.getenv("COOKIES_PATH", "/runpod-volume/llave.txt")

# Caché de modelos HuggingFace (pyannote) dentro del contenedor para no re-descargar
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# ── Arranque ──────────────────────────────────────────────────────────────────
# -u: sin buffer en stdout → logs visibles en tiempo real en RunPod
CMD ["python", "-u", "handler.py"]
