"""
KLYPO API — FastAPI backend v2 (RunPod Serverless)

El procesamiento ocurre en RunPod, no en este servidor.
Este backend actua como proxy entre el frontend y la API de RunPod.

Flujo:
    POST /generar         → envia job a RunPod /run, guarda runpod_job_id, devuelve job_id
    GET  /estado/{job_id} → consulta RunPod /status, mapea estado, devuelve clips con URLs R2
    GET  /download/{path} → sirve clips locales (solo modo desarrollo local)
    GET  /                → health check + modo activo

Correr en local (apunta a RunPod de todas formas):
    uvicorn api:app --reload --port 8000

Variables de entorno requeridas (.env):
    RUNPOD_API_KEY      - API key de RunPod (https://www.runpod.io/console/user/settings)
    RUNPOD_ENDPOINT_ID  - ID del endpoint serverless (aparece en la URL del endpoint en RunPod)
"""

import os
import sys
import traceback

import requests as _req
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "modulos_virales"))

RUNPOD_API_KEY     = os.environ.get("RUNPOD_API_KEY", "").strip()
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
RUNPOD_BASE        = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="KLYPO API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Jobs en memoria ───────────────────────────────────────────────────────────
# Estructura de cada job:
# {
#   "estado":         "en_cola" | "procesando" | "listo" | "error"
#   "progreso":       str   — mensaje legible para el frontend
#   "clips":          [str] — URLs de R2 (en prod) o paths /download/... (dev)
#   "clips_detalle":  [{"nombre":..., "url":..., "local":...}]  (solo RunPod)
#   "error":          str
#   "runpod_job_id":  str   — ID del job en RunPod
# }
jobs: dict = {}

# ── Modelo de entrada ─────────────────────────────────────────────────────────
class SolicitudClip(BaseModel):
    url:        str
    modo:       str  = "viral"    # viral | podcast
    fuente_sub: str  = "Anton"    # Anton | Arial | Montserrat | BebasNeue | Poppins
    mayusculas: bool = False
    modo_sub:   str  = "bloques"  # bloques | karaoke | none


# ── Helpers RunPod ────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type":  "application/json",
    }


# Mapa de estados RunPod → estados internos que usa el frontend
_ESTADO = {
    "IN_QUEUE":    "en_cola",
    "IN_PROGRESS": "procesando",
    "COMPLETED":   "listo",
    "FAILED":      "error",
    "CANCELLED":   "error",
    "TIMED_OUT":   "error",
}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    configurado = bool(RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID)
    return {
        "status":       "KLYPO API funcionando ⚡",
        "modo":         "runpod" if configurado else "sin-configurar",
        "jobs_activos": len(jobs),
    }


@app.post("/generar")
def generar(req: SolicitudClip):
    """
    Envia el job a RunPod Serverless y devuelve el job_id interno.
    El frontend usa ese job_id para hacer polling con /estado/{job_id}.
    """
    if not req.url.strip():
        raise HTTPException(400, "URL vacia")

    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        raise HTTPException(
            503,
            "RunPod no configurado. "
            "Agrega RUNPOD_API_KEY y RUNPOD_ENDPOINT_ID en el archivo .env",
        )

    # Payload que espera el handler.py en el worker de RunPod
    payload = {
        "input": {
            "url":        req.url.strip(),
            "modo":       req.modo,
            "fuente_sub": req.fuente_sub,
            "mayusculas": req.mayusculas,
            "modo_sub":   req.modo_sub,
        }
    }

    try:
        resp = _req.post(
            f"{RUNPOD_BASE}/run",
            json    = payload,
            headers = _headers(),
            timeout = 30,
        )
        resp.raise_for_status()
        data = resp.json()
    except _req.exceptions.RequestException as e:
        raise HTTPException(502, f"Error conectando con RunPod: {e}")

    runpod_job_id = data.get("id")
    if not runpod_job_id:
        raise HTTPException(502, f"RunPod no devolvio job ID: {data}")

    # Usar el ID de RunPod directamente como job_id publico.
    # Asi /estado/{job_id} puede consultar RunPod aunque el dict en memoria
    # se haya vaciado por un --reload o reinicio del servidor.
    job_id = runpod_job_id
    jobs[job_id] = {
        "estado":        "en_cola",
        "progreso":      "🕐 Job enviado a RunPod — en cola...",
        "clips":         [],
        "clips_detalle": [],
        "error":         "",
    }

    print(f"✅ Job creado: RunPod={job_id}")
    return {"job_id": job_id, "mensaje": "Procesamiento enviado a RunPod"}


@app.get("/estado/{job_id}")
def estado(job_id: str):
    """
    Polling del estado de un job.
    Cada llamada consulta RunPod en tiempo real (excepto si ya terminó — cachea resultado).
    El frontend llama esto cada ~5s hasta que estado == 'listo' o 'error'.

    Respuesta:
      {
        "estado":   "en_cola" | "procesando" | "listo" | "error",
        "progreso": "mensaje legible",
        "clips":    ["https://r2.url/clip1.mp4", ...],   <- URLs directas de descarga
        "error":    "mensaje de error si aplica"
      }
    """
    job = jobs.get(job_id)

    # Si ya terminó con resultado cacheado, devolver sin llamar a RunPod
    if job and job["estado"] in ("listo", "error"):
        return job

    # Consultar RunPod directamente — job_id ES el runpod_job_id.
    # Funciona aunque jobs este vacio (--reload, reinicio del servidor).
    try:
        resp = _req.get(
            f"{RUNPOD_BASE}/status/{job_id}",
            headers = _headers(),
            timeout = 15,
        )
        resp.raise_for_status()
        data = resp.json()
    except _req.exceptions.RequestException as e:
        if job:
            # Error de red transitorio: devolver ultimo estado conocido del cache
            print(f"⚠️  RunPod status error (job {job_id}): {e}")
            return job
        raise HTTPException(503, f"No se pudo consultar RunPod: {e}")

    runpod_status = data.get("status", "")
    if not runpod_status:
        raise HTTPException(404, f"Job '{job_id}' no existe en RunPod")

    # Reconstruir entrada en cache si se perdio por reinicio del servidor
    if job is None:
        job = {
            "estado":        "en_cola",
            "progreso":      "",
            "clips":         [],
            "clips_detalle": [],
            "error":         "",
        }
        jobs[job_id] = job

    job["estado"] = _ESTADO.get(runpod_status, "en_cola")

    if runpod_status == "IN_QUEUE":
        job["progreso"] = "🕐 En cola en RunPod..."

    elif runpod_status == "IN_PROGRESS":
        job["progreso"] = "⚙️ Procesando: descarga → transcripcion → deteccion → clips..."

    elif runpod_status == "COMPLETED":
        output = data.get("output") or {}

        if "error" in output:
            # El handler de RunPod devolvio {"error": "..."} en su output
            job["estado"]   = "error"
            job["error"]    = output["error"]
            job["progreso"] = f"❌ Error en RunPod: {output['error']}"

        else:
            clips_raw = output.get("clips", [])
            # clips_raw: [{"nombre": "...", "url": "https://r2...", "local": "..."}, ...]

            # URLs de R2 como lista simple — el frontend las usa directamente como links
            urls = [c["url"] for c in clips_raw if c.get("url")]
            n    = len(urls)

            job["clips"]         = urls
            job["clips_detalle"] = clips_raw   # detalle completo (nombre + url + local)
            job["estado"]        = "listo"
            job["progreso"]      = (
                f"✅ {n} clip{'s' if n != 1 else ''} "
                f"listo{'s' if n != 1 else ''} para descargar"
            )

    elif runpod_status in ("FAILED", "CANCELLED", "TIMED_OUT"):
        # Intentar extraer mensaje de error de donde RunPod lo ponga
        error_msg = (
            data.get("error")
            or (data.get("output") or {}).get("error")
            or f"RunPod: {runpod_status}"
        )
        job["estado"]   = "error"
        job["error"]    = error_msg
        job["progreso"] = f"❌ {error_msg}"

    return job


@app.get("/download/{path:path}")
def download(path: str):
    """
    Sirve clips locales (modo desarrollo).
    En produccion con RunPod los clips se descargan directamente desde R2 —
    este endpoint no se usa para esos jobs.
    """
    full_path = os.path.abspath(os.path.join(BASE_DIR, path))

    if not full_path.startswith(BASE_DIR):
        raise HTTPException(403, "Acceso denegado")
    if not full_path.endswith(".mp4"):
        raise HTTPException(403, "Solo se permiten archivos .mp4")
    if not os.path.exists(full_path):
        raise HTTPException(404, "Clip no encontrado")

    filename = os.path.basename(full_path)
    return FileResponse(
        full_path,
        media_type = "video/mp4",
        filename   = filename,
        headers    = {"Content-Disposition": f'attachment; filename="{filename}"'},
    )
