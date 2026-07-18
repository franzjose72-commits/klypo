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
from typing import Optional

import time as _time

import requests as _req
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
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

SUPABASE_URL         = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()

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


# ── Supabase: verificación de token + helpers REST ────────────────────────────

_token_cache: dict[str, tuple[str, float]] = {}  # {token: (user_id, expira_unix)}
_TOKEN_CACHE_TTL = 60  # segundos


def _verificar_token(authorization: str | None) -> str:
    """
    Verifica el JWT del usuario preguntando a Supabase /auth/v1/user.
    Cachea el resultado 60s para no golpear Supabase en reenvíos rápidos.
    Devuelve el user_id (uuid) o lanza HTTPException.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token de autenticación requerido")

    token = authorization[7:]
    ahora = _time.time()

    if token in _token_cache:
        user_id, expira = _token_cache[token]
        if ahora < expira:
            return user_id
        del _token_cache[token]

    try:
        resp = _req.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey":        SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {token}",
            },
            timeout=8,
        )
    except _req.exceptions.Timeout:
        raise HTTPException(503, "Servicio de autenticación lento — intenta de nuevo en unos segundos")
    except _req.exceptions.RequestException as e:
        raise HTTPException(503, f"No se pudo verificar la sesión: {e}")

    if resp.status_code == 401:
        raise HTTPException(401, "Sesión expirada — vuelve a iniciar sesión")
    if not resp.ok:
        raise HTTPException(502, f"Error verificando sesión ({resp.status_code})")

    user_id = resp.json().get("id", "")
    if not user_id:
        raise HTTPException(401, "Token inválido: no contiene ID de usuario")

    _token_cache[token] = (user_id, ahora + _TOKEN_CACHE_TTL)
    if len(_token_cache) > 500:
        _token_cache.clear()

    return user_id


def _supabase_rpc(funcion: str, params: dict) -> dict:
    resp = _req.post(
        f"{SUPABASE_URL}/rest/v1/rpc/{funcion}",
        json=params,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey":        SUPABASE_SERVICE_KEY,
            "Content-Type":  "application/json",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _supabase_insert(tabla: str, datos: dict) -> dict:
    resp = _req.post(
        f"{SUPABASE_URL}/rest/v1/{tabla}",
        json=datos,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey":        SUPABASE_SERVICE_KEY,
            "Content-Type":  "application/json",
            "Prefer":        "return=representation",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()[0]


def _supabase_patch(tabla: str, filtros: dict, datos: dict) -> None:
    params = "&".join(f"{k}=eq.{v}" for k, v in filtros.items())
    _req.patch(
        f"{SUPABASE_URL}/rest/v1/{tabla}?{params}",
        json=datos,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey":        SUPABASE_SERVICE_KEY,
            "Content-Type":  "application/json",
        },
        timeout=10,
    )


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
def generar(req: SolicitudClip, authorization: str | None = Header(default=None)):
    """
    Envia el job a RunPod Serverless y devuelve el job_id interno.
    El frontend usa ese job_id para hacer polling con /estado/{job_id}.
    """
    if not req.url.strip():
        raise HTTPException(400, "URL vacia")

    # Verificar identidad (sin bloquear todavía — créditos llegan en Paso 6)
    user_id = None
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        try:
            user_id = _verificar_token(authorization)
            print(f"🪪 /generar — user_id={user_id}", flush=True)
        except HTTPException as e:
            print(f"⚠️  /generar — token inválido o ausente: {e.detail}", flush=True)

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

    # [PASO 4] Backend crea el proyecto en Supabase (elimina clips huérfanos)
    proyecto_id = None
    if user_id and SUPABASE_URL and SUPABASE_SERVICE_KEY:
        try:
            proyecto = _supabase_insert("proyectos", {
                "job_id":  job_id,
                "user_id": user_id,
                "modo":    req.modo,
                "clips":   [],
            })
            proyecto_id = proyecto.get("id")
            print(f"📁 Proyecto creado en Supabase: {proyecto_id}", flush=True)
        except Exception as e:
            print(f"⚠️  No se pudo crear el proyecto en Supabase: {e}", flush=True)
            # No bloqueamos — el job sigue, el front verá clips aunque no queden en historial

    print(f"✅ Job creado: RunPod={job_id} | proyecto={proyecto_id}", flush=True)
    return {"job_id": job_id, "proyecto_id": proyecto_id, "mensaje": "Procesamiento enviado a RunPod"}


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

    except _req.exceptions.HTTPError as http_err:
        # Error HTTP de RunPod (404, 500, etc.) — distinto de error de red
        code = http_err.response.status_code if http_err.response is not None else 0
        if code == 404:
            # RunPod ya no tiene el job: expiró después de completarse.
            if job and job.get("clips"):
                # Tenemos los clips de un polling COMPLETED anterior → devolver listo
                job["estado"]   = "listo"
                job["progreso"] = f"✅ {len(job['clips'])} clip(s) listos para descargar"
                return job
            # Job expiró sin que hubiéramos capturado el resultado
            err = "Job expirado en RunPod sin resultado disponible. Vuelve a enviar el video."
            if job:
                job["estado"]   = "error"
                job["error"]    = err
                job["progreso"] = f"❌ {err}"
                return job
            raise HTTPException(404, err)
        # Otro error HTTP (500, etc.) — devolver último estado o propagar
        print(f"⚠️  RunPod HTTP {code} (job {job_id}): {http_err}")
        if job:
            return job
        raise HTTPException(502, f"Error HTTP de RunPod ({code})")

    except _req.exceptions.RequestException as e:
        # Error de red transitorio (timeout, conexión): devolver último estado conocido
        print(f"⚠️  RunPod status error (job {job_id}): {e}")
        if job:
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
        # /stream es best-effort para mostrar progreso en vivo.
        # Timeout corto (5s) para no bloquear el polling cada 5s del frontend.
        # Si falla → silencio; los clips finales llegan garantizados al COMPLETED.
        try:
            s_resp = _req.get(
                f"{RUNPOD_BASE}/stream/{job_id}",
                headers=_headers(),
                timeout=5,
            )
            if s_resp.ok:
                stream_raw = s_resp.json()
                items  = stream_raw if isinstance(stream_raw, list) else stream_raw.get("stream", [])
                urls_s = set(job["clips"])
                for si in items:
                    if not isinstance(si, dict):
                        continue
                    # Formato A: {"output": {...}, "status": "..."}
                    # Formato B: yield crudo {"clips": [...]}
                    raw = si.get("output") if "output" in si else si
                    if not isinstance(raw, dict):
                        raw = si
                    for clip in raw.get("clips", []):
                        url = clip.get("url") if isinstance(clip, dict) else None
                        if url and url not in urls_s:
                            urls_s.add(url)
                            job["clips"].append(url)
                            job["clips_detalle"].append(clip)
        except Exception:
            pass  # best-effort — timeout o error no interrumpe el polling
        n = len(job["clips"])
        job["progreso"] = (
            f"⚙️ Generando clips... {n} clip{'s' if n != 1 else ''} "
            f"{'listos' if n != 1 else 'listo'} hasta ahora"
            if n else "⚙️ Procesando: descarga → transcripcion → deteccion → clips..."
        )

    elif runpod_status == "COMPLETED":
        # El último yield del generator tiene TODOS los clips en clips_resultado.
        # data.get("output") viene del /status (ya fetched), no del /stream.
        # Si RunPod lo devuelve como lista de yields → escaneamos todos.
        # Si lo devuelve como el dict del último yield → lo usamos directo.
        raw_output = data.get("output") or {}
        print(f"🔍 COMPLETED raw_output type={type(raw_output).__name__} "
              f"len={len(raw_output) if isinstance(raw_output, (list, dict)) else '?'}")

        output_items = raw_output if isinstance(raw_output, list) else [raw_output]

        urls_acum = set(job["clips"])  # clips parciales ya recogidos del /stream/
        error_msg = ""
        for item in output_items:
            if not isinstance(item, dict):
                continue
            raw = item.get("output") if "output" in item else item
            if not isinstance(raw, dict):
                raw = item
            if "error" in raw and not error_msg:
                error_msg = raw["error"]
            for c in raw.get("clips", []):
                url = c.get("url") if isinstance(c, dict) else None
                if url and url not in urls_acum:
                    urls_acum.add(url)
                    job["clips"].append(url)
                    job["clips_detalle"].append(c)

        n = len(job["clips"])
        print(f"🔍 COMPLETED: {n} clips en job tras escanear output "
              f"({len(output_items)} items, error_msg={error_msg!r})")

        if n > 0:
            job["estado"]   = "listo"
            job["progreso"] = (
                f"✅ {n} clip{'s' if n != 1 else ''} "
                f"{'listos' if n != 1 else 'listo'} para descargar"
            )
            # [PASO 5] Backend vincula clips al proyecto en Supabase (idempotente)
            # Usa job_id como filtro para funcionar aunque jobs[] esté vacío por reinicio.
            if SUPABASE_URL and SUPABASE_SERVICE_KEY:
                try:
                    _supabase_patch(
                        "proyectos",
                        {"job_id": job_id},
                        {"clips": job["clips"]},
                    )
                    print(f"📎 {n} clips vinculados a proyecto (job {job_id})", flush=True)
                except Exception as e:
                    print(f"⚠️  No se pudo vincular clips en Supabase (job {job_id}): {e}", flush=True)
        else:
            job["estado"]   = "error"
            job["error"]    = error_msg or "No se generaron clips"
            job["progreso"] = f"❌ {job['error']}"

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


@app.get("/presigned-upload")
def presigned_upload(filename: str = Query(..., description="Nombre del archivo a subir")):
    """
    Genera una presigned PUT URL para que el navegador suba un video DIRECTAMENTE a R2.
    El archivo nunca pasa por este servidor — va navegador → R2.

    Flujo:
      1. Frontend llama GET /presigned-upload?filename=mi_video.mp4
      2. Recibe { upload_url, public_url, key }
      3. Frontend hace PUT a upload_url con el archivo (XHR con progress)
      4. Frontend llama POST /generar con public_url como campo 'url'
      5. RunPod descarga el video de R2 directamente (HTTP GET simple)
    """
    if not filename.strip():
        raise HTTPException(400, "Falta el nombre de archivo")

    from subir_r2 import generar_presigned_put
    resultado = generar_presigned_put(filename.strip())

    if not resultado:
        raise HTTPException(503, "R2 no está configurado (faltan credenciales en .env)")

    if not resultado.get("public_url"):
        raise HTTPException(
            503,
            "R2_PUBLIC_URL no está configurado. "
            "RunPod necesita esa URL para descargar el video. "
            "Agrégala en .env: R2_PUBLIC_URL=https://pub-XXXX.r2.dev"
        )

    return resultado   # {put_url, public_url, key}


@app.get("/clips-parciales/{job_id}")
def clips_parciales(job_id: str):
    """
    Lista los clips ya subidos a R2 con el prefijo job_id/.
    El frontend lo llama cada 5s durante procesamiento para mostrar clips en tiempo real,
    sin depender del /stream de RunPod (que da timeout cuando el worker está ocupado).
    Siempre devuelve {"clips": []} ante cualquier error — nunca rompe el polling.
    """
    if not job_id or not job_id.strip():
        return {"clips": []}
    try:
        import boto3.session as b3s
        from botocore.config import Config as _Cfg
        from subir_r2 import _limpiar_proxy, _restaurar_proxy

        access_key = os.environ.get("R2_ACCESS_KEY_ID",     "").strip()
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
        endpoint   = os.environ.get("R2_ENDPOINT",          "").strip()
        bucket     = os.environ.get("R2_BUCKET", "klypo-clips").strip()
        r2_public  = os.environ.get("R2_PUBLIC_URL", "").strip().rstrip("/")

        if not all([access_key, secret_key, endpoint, r2_public]):
            return {"clips": []}

        guardadas = _limpiar_proxy()
        try:
            s3 = b3s.Session().client(
                "s3",
                endpoint_url          = endpoint,
                aws_access_key_id     = access_key,
                aws_secret_access_key = secret_key,
                region_name           = "auto",
                config                = _Cfg(
                    signature_version = "s3v4",
                    proxies           = {"http": "", "https": ""},
                    s3                = {"addressing_style": "path"},
                ),
            )
            resp  = s3.list_objects_v2(Bucket=bucket, Prefix=f"{job_id}/")
            clips = [
                f"{r2_public}/{obj['Key']}"
                for obj in resp.get("Contents", [])
                if obj["Key"].lower().endswith(".mp4")
            ]
            return {"clips": clips}
        finally:
            _restaurar_proxy(guardadas)

    except Exception as e:
        print(f"⚠️ /clips-parciales/{job_id}: {e}")
        return {"clips": []}


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
