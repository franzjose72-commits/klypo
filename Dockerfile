# KLYPO - RunPod Serverless
# CUDA 12.8 + PyTorch 2.7 -> soporta GPUs Blackwell sm_120 (RTX PRO 6000, B200, B100)
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Sistema: Python 3.11 + ffmpeg + OpenCV + fuentes + utilidades
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    fontconfig \
    git \
    curl \
    unzip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Deno: runtime JS que yt-dlp necesita para resolver el challenge "n" de YouTube
RUN curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
    -o /tmp/deno.zip \
    && unzip /tmp/deno.zip -d /usr/local/bin/ \
    && chmod +x /usr/local/bin/deno \
    && rm /tmp/deno.zip \
    && deno --version

WORKDIR /app

# Dependencias Python (requirements primero para que torch no sea pisado por pyannote)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch 2.7 cu128 al final con force-reinstall para garantizar sm_120 (Blackwell)
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.7.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Verificacion: confirma que sm_120 esta en la lista
RUN python -c "import torch; print('torch', torch.__version__); print('arch_list', torch.cuda.get_arch_list())"

# Codigo de la aplicacion
COPY modulos_virales/ ./modulos_virales/
COPY fonts/           ./fonts/
COPY handler.py       .

# Cache HuggingFace (montar Network Volume en /app/.cache en RunPod para persistir modelos)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

CMD ["python", "-u", "handler.py"]
