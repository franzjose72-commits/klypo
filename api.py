"""
KLYPO API — FastAPI backend
Conecta el frontend web con el motor de clips virales.

Correr en local:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /generar          → inicia un job, devuelve job_id
    GET  /estado/{job_id}  → polling del estado del job
    GET  /download/{path}  → descarga un clip .mp4
    GET  /                 → health check
"""

import os
import sys
import uuid
import threading
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Path para que Python encuentre modulos_virales/ ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "modulos_virales"))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="KLYPO API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # en producción: reemplaza con tu dominio
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Jobs en memoria (para RunPod usar Redis) ──────────────────────────────────
jobs: dict = {}
# Estructura de cada job:
# {
#   "estado":   "en_cola" | "procesando" | "listo" | "error",
#   "progreso": str (mensaje legible para mostrar al usuario),
#   "clips":    ["/download/CLIPS_VIRAL_V78/KLYPO_V78_01_...mp4", ...],
#   "error":    str,
# }

# ── Modelos de entrada ────────────────────────────────────────────────────────
class SolicitudClip(BaseModel):
    url:        str
    fuente_sub: str  = "Anton"     # Anton | Arial | Montserrat | BebasNeue | Poppins
    mayusculas: bool = False
    modo_sub:   str  = "bloques"   # bloques | karaoke | none


# ── Worker en hilo separado ───────────────────────────────────────────────────
def _worker(job_id: str, req: SolicitudClip):
    try:
        jobs[job_id]["estado"]   = "procesando"
        jobs[job_id]["progreso"] = "⬇️ Descargando video de YouTube..."

        from motor_viral import procesar_viral

        def _log_progreso(msg: str):
            jobs[job_id]["progreso"] = msg

        def _clip_listo(ruta: str):
            rel = os.path.relpath(ruta, BASE_DIR).replace("\\", "/")
            jobs[job_id]["clips"].append(f"/download/{rel}")

        _log_progreso("🎙️ Transcribiendo audio con AssemblyAI...")
        rutas = procesar_viral(
            req.url,
            fuente_sub   = req.fuente_sub,
            mayusculas   = req.mayusculas,
            modo_sub     = req.modo_sub,
            progreso_cb  = _log_progreso,
            clip_listo_cb = _clip_listo,
        )

        n = len(jobs[job_id]["clips"])
        jobs[job_id]["estado"]   = "listo"
        jobs[job_id]["progreso"] = f"✅ {n} clip{'s' if n != 1 else ''} listo{'s' if n != 1 else ''} para descargar"

    except Exception as e:
        jobs[job_id]["estado"]   = "error"
        jobs[job_id]["error"]    = str(e)
        jobs[job_id]["progreso"] = f"❌ Error: {e}"
        traceback.print_exc()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "KLYPO API funcionando ⚡", "jobs_activos": len(jobs)}


@app.post("/generar")
def generar(req: SolicitudClip):
    """
    Inicia el procesamiento en background.
    Devuelve job_id para hacer polling con /estado/{job_id}.
    """
    if not req.url.strip():
        raise HTTPException(400, "URL vacía")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "estado":   "en_cola",
        "progreso": "🕐 En cola...",
        "clips":    [],
        "error":    "",
    }

    t = threading.Thread(target=_worker, args=(job_id, req), daemon=True)
    t.start()

    return {"job_id": job_id, "mensaje": "Procesamiento iniciado"}


@app.get("/estado/{job_id}")
def estado(job_id: str):
    """
    Polling del estado de un job.
    El frontend llama esto cada 5s hasta que estado == 'listo' o 'error'.
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job no encontrado")
    return jobs[job_id]


@app.get("/download/{path:path}")
def download(path: str):
    """
    Sirve un clip .mp4 generado por KLYPO.
    Solo permite acceso a archivos .mp4 dentro del proyecto.
    """
    full_path = os.path.abspath(os.path.join(BASE_DIR, path))

    # Seguridad: no permitir path traversal (../../etc)
    if not full_path.startswith(BASE_DIR):
        raise HTTPException(403, "Acceso denegado")
    if not full_path.endswith(".mp4"):
        raise HTTPException(403, "Solo se permiten archivos .mp4")
    if not os.path.exists(full_path):
        raise HTTPException(404, "Clip no encontrado")

    filename = os.path.basename(full_path)
    return FileResponse(
        full_path,
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
