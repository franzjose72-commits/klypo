"""
KLYPO — RunPod Serverless Handler

Recibe un job de RunPod, procesa el video y devuelve las URLs de los clips en R2.

Input esperado (event["input"]):
    url        : str   — URL de YouTube o ruta local al video
    fuente_sub : str   — Anton | Arial | Montserrat | BebasNeue | Poppins  (default: Anton)
    mayusculas : bool  — True = MAYUSCULAS en subtitulos  (default: False)
    modo_sub   : str   — bloques | karaoke | none          (default: bloques)
    modo       : str   — "viral" | "podcast"               (default: viral)

Limites:
    MAX_CLIPS           = 15    clips devueltos como maximo
    MAX_DUR_VIRAL_SEG   = 3600  segundos (1 hora)  — modo viral,   rechaza antes de descargar
    MAX_DUR_PODCAST_SEG = 7200  segundos (2 horas) — modo podcast, rechaza antes de descargar
"""

import os
import sys
import subprocess
import traceback

import requests
import runpod

# modulos_virales/ está al mismo nivel que handler.py dentro del contenedor
_RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_RAIZ, "modulos_virales"))

MAX_CLIPS             = 15
MAX_DUR_VIRAL_SEG     = 3600   # 1 hora  — modo viral
MAX_DUR_PODCAST_SEG   = 3600   # 1 hora  — modo podcast (límite Groq free plan)

# ── Apify (descarga YouTube en 1080p) ─────────────────────────────────────────
# Pon tu token en RunPod → endpoint → Environment Variables → APIFY_TOKEN
APIFY_TOKEN  = os.environ.get("APIFY_TOKEN", "").strip()
_APIFY_ACTOR = "UUhJDfKJT2SsXdclR"   # streamers/youtube-video-downloader
_APIFY_BASE  = "https://api.apify.com/v2"

# ── Cookies de YouTube ────────────────────────────────────────────────────────

def _setup_cookies() -> str:
    """
    Devuelve la ruta al archivo de cookies que usará yt-dlp.

    Prioridad:
      1. YOUTUBE_COOKIES (variable de entorno) → escribe llave.txt en la raíz del proyecto.
         Usar en RunPod: pega todo el contenido del archivo en esa variable de entorno.
      2. llave.txt ya existe en el disco → úsalo directamente (modo desarrollo local).
      3. Ninguno → devuelve la ruta esperada de todas formas; yt-dlp fallará con un error claro.
    """
    cookies_path = os.path.join(_RAIZ, "llave.txt")
    env_cookies  = os.environ.get("YOUTUBE_COOKIES", "").strip()

    if env_cookies:
        with open(cookies_path, "w", encoding="utf-8") as f:
            f.write(env_cookies)
        print(f"🔑 Cookies escritas desde YOUTUBE_COOKIES → {cookies_path} ({len(env_cookies)} bytes)")
    elif os.path.exists(cookies_path):
        print(f"🔑 Usando llave.txt local: {cookies_path} ({os.path.getsize(cookies_path)} bytes)")
    else:
        print("⚠️  No hay cookies: define YOUTUBE_COOKIES en RunPod o crea llave.txt localmente")

    return cookies_path


# Ejecutar al arrancar el worker (antes de cualquier job)
_setup_cookies()


def _precargar_modulos_podcast():
    """
    Importa camara y subtitulos durante el warmup del worker (ANTES de recibir jobs).
    Así MediaPipe, torch y la verificación de fuentes ocurren fuera del executionTimeout.
    """
    try:
        import camara       # inicializa MediaPipe FaceDetection + carga torch
        print("✅ [preload] camara OK")
    except Exception as e:
        print(f"⚠️ [preload] camara: {e}")
    try:
        import subtitulos   # verifica fuentes locales (sin red)
        print("✅ [preload] subtitulos OK")
    except Exception as e:
        print(f"⚠️ [preload] subtitulos: {e}")

_precargar_modulos_podcast()


# ── Descarga directa (para archivos subidos por el usuario, no YouTube) ───────

def _es_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def _descargar_directo(url: str, dest: str) -> str:
    """
    Descarga un archivo de video desde una URL directa (R2, CDN, etc.) usando requests.
    Se usa cuando el usuario sube su propio video en vez de pegar un link de YouTube.
    Devuelve la ruta local del archivo descargado.
    """
    print(f"⬇️  Descargando archivo directo (no YouTube): {url[:80]}...")
    resp = requests.get(url, stream=True, timeout=(30, 600))
    resp.raise_for_status()

    total      = int(resp.headers.get("content-length", 0))
    descargado = 0
    ultimo_log = 0

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
            if chunk:
                f.write(chunk)
                descargado += len(chunk)
                if total and descargado - ultimo_log >= 50 * 1024 * 1024:  # log cada 50 MB
                    pct = descargado / total * 100
                    print(f"   📥 {pct:.0f}%  {descargado // 1_048_576} MB / {total // 1_048_576} MB")
                    ultimo_log = descargado

    print(f"✅ Descarga directa completa: {descargado // 1_048_576} MB → {dest}")
    return dest


def _descargar_via_apify(youtube_url: str, dest: str) -> str:
    """
    Descarga un video de YouTube via Apify (streamers/youtube-video-downloader).
    Devuelve la ruta local del archivo descargado.
    Lanza RuntimeError con mensaje en español si algo falla.

    Flujo:
      1. POST /runs → arranca el actor y recibe run_id
      2. Polling GET /actor-runs/{run_id} cada 15s hasta SUCCEEDED o FAILED (max 20 min)
      3. GET /actor-runs/{run_id}/dataset/items → extrae downloadedFileUrl
      4. Descarga el MP4 con _descargar_directo() (función ya existente)
    """
    import time as _time

    if not APIFY_TOKEN:
        raise RuntimeError(
            "APIFY_TOKEN no está configurado. "
            "Agrégalo en RunPod → tu endpoint → Environment Variables → APIFY_TOKEN."
        )

    print(f"🎬 Apify: solicitando descarga de {youtube_url[:70]}...")

    # 1. Arrancar el run del actor
    try:
        run_resp = requests.post(
            f"{_APIFY_BASE}/acts/{_APIFY_ACTOR}/runs",
            params  = {"token": APIFY_TOKEN},
            json    = {
                "videos":                [{"url": youtube_url}],
                "preferredQuality":      "1080p",
                "preferredFormat":       "mp4",
                "storeInKVStore":        False,
                "filenameTemplateParts": ["title"],
            },
            timeout = 30,
        )
        run_resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"No se pudo conectar con Apify para iniciar la descarga: {e}")

    run_id = run_resp.json().get("data", {}).get("id", "")
    if not run_id:
        raise RuntimeError(f"Apify no devolvió un ID de run. Respuesta: {run_resp.text[:200]}")
    print(f"   ✅ Run iniciado: {run_id}")

    # 2. Polling: esperar hasta SUCCEEDED o FAILED (max 80 intentos × 15s = 20 min)
    status = ""
    for intento in range(1, 81):
        _time.sleep(15)
        try:
            st_resp = requests.get(
                f"{_APIFY_BASE}/actor-runs/{run_id}",
                params  = {"token": APIFY_TOKEN},
                timeout = 30,
            )
            st_resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ Apify polling error (intento {intento}): {e}")
            continue

        status = st_resp.json().get("data", {}).get("status", "")
        print(f"   ⏳ Apify [{intento}/80]: {status}")

        if status == "SUCCEEDED":
            break
        if status in ("FAILED", "TIMED-OUT", "ABORTED"):
            raise RuntimeError(
                f"Apify no pudo descargar el video (estado: {status}). "
                "El video puede ser privado, age-restricted, o la URL no es válida."
            )
    else:
        raise RuntimeError("Timeout de 20 minutos esperando que Apify termine la descarga.")

    # 3. Obtener dataset items para extraer downloadedFileUrl
    try:
        items_resp = requests.get(
            f"{_APIFY_BASE}/actor-runs/{run_id}/dataset/items",
            params  = {"token": APIFY_TOKEN},
            timeout = 30,
        )
        items_resp.raise_for_status()
        items = items_resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"No se pudo obtener el resultado de Apify: {e}")

    if not items or not isinstance(items, list):
        raise RuntimeError("Apify completó pero el dataset está vacío.")

    file_url = items[0].get("downloadedFileUrl", "")
    if not file_url:
        raise RuntimeError(
            f"Apify no devolvió 'downloadedFileUrl' en el resultado. "
            f"Item recibido: {str(items[0])[:200]}"
        )

    print(f"   🔗 Apify URL: {file_url[:80]}...")

    # 4. Descargar el MP4 usando la función ya existente
    return _descargar_directo(file_url, dest)


def _duracion_video(url: str) -> float | None:
    """
    Obtiene la duración en segundos SIN descargar el video completo.
    - Para URLs de YouTube: usa yt-dlp --print duration (solo metadata).
    - Para archivos locales: usa ffprobe.
    Devuelve None si no se puede determinar (no bloquea el job).
    """
    try:
        if url.startswith("http"):
            cmd = ["yt-dlp", "--skip-download", "--print", "duration", "--no-warnings"]
            proxy_user = os.environ.get("PROXY_USER", "").strip()
            proxy_pass = os.environ.get("PROXY_PASS", "").strip()
            if proxy_user and proxy_pass:
                cmd += ["--proxy", f"http://{proxy_user}:{proxy_pass}@gw.dataimpulse.com:823"]
            cmd.append(url)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                return float(r.stdout.strip())
        else:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                url,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                return float(r.stdout.strip())
    except Exception as e:
        print(f"⚠️ No se pudo obtener duración: {e}")
    return None


def handler(event):
    inp = event.get("input", {})

    url        = inp.get("url",        "").strip()
    fuente_sub = inp.get("fuente_sub", "Anton").strip()
    mayusculas = bool(inp.get("mayusculas", False))
    modo_sub   = inp.get("modo_sub",   "bloques").strip().lower()
    modo       = inp.get("modo",       "viral").strip().lower()

    # Normalizar variantes del modo por si el frontend manda valores distintos
    _MODO_ALIAS = {
        "viral":        "viral",
        "clipsvirales": "viral",
        "clips_virales":"viral",
        "clipsviral":   "viral",
        "clips":        "viral",
        "podcast":      "podcast",
    }
    modo = _MODO_ALIAS.get(modo, modo)

    # ID del job en RunPod — se usa como prefijo en R2 para aislar clips por job
    job_id_prefix = event.get("id", "").strip()

    # ── Validación de entrada ─────────────────────────────────────────────────
    if not url:
        return {"error": "Falta el campo 'url' en el input"}

    # ── Límite de duración según modo (ANTES de descargar) ───────────────────────
    _limite_dur = MAX_DUR_VIRAL_SEG if modo == "viral" else MAX_DUR_PODCAST_SEG
    dur = _duracion_video(url)
    if dur is not None and dur > _limite_dur:
        dur_min    = int(dur // 60)
        limite_min = _limite_dur // 60
        _modo_es   = 'Clips Virales' if modo == 'viral' else 'Podcast'
        _modo_en   = 'Viral Clips'   if modo == 'viral' else 'Podcast'
        return {
            "error_code": "VIDEO_TOO_LONG",
            "error": (
                f"El modo {_modo_es} acepta videos de hasta {limite_min} min "
                f"(este video: {dur_min} min). Recorta el video o usa un segmento más corto. "
                f"/ {_modo_en} mode accepts up to {limite_min} min "
                f"(this video: {dur_min} min). Please trim or use a shorter segment."
            ),
            "error_data": {
                "dur_min":    dur_min,
                "limite_min": limite_min,
                "modo":       modo,
            },
        }

    # ── Pre-descarga: convierte cualquier URL a ruta local antes de procesar ─────
    # Ambos modos (viral y podcast) aceptan rutas locales sin cambio alguno.
    if url.startswith("http") and _es_youtube(url):
        # YouTube: descarga via Apify en 1080p (reemplaza yt-dlp que daba 480p)
        dest_apify = os.path.join(_RAIZ, "video_apify.mp4")
        try:
            url = _descargar_via_apify(url, dest_apify)
        except Exception as e:
            print(f"❌ Apify error: {e}")
            return {"error": str(e)}

    elif url.startswith("http"):
        # Video subido por el usuario (R2, CDN, etc.): descarga directa sin cambios
        dest_directo = os.path.join(_RAIZ, "video_directo.mp4")
        try:
            url = _descargar_directo(url, dest_directo)
        except Exception as e:
            print(f"❌ Error descargando archivo directo: {e}")
            return {"error": f"No se pudo descargar el archivo de video: {e}"}

    fuentes_validas = {"Anton", "Arial", "Montserrat", "BebasNeue", "Poppins"}
    if fuente_sub not in fuentes_validas:
        fuente_sub = "Anton"

    if modo_sub not in {"bloques", "karaoke", "none"}:
        modo_sub = "bloques"

    print(f"🚀 KLYPO handler — modo={modo} | url={url[:60]} | fuente={fuente_sub} | modo_sub={modo_sub}")

    # ── Ruteo de modos ────────────────────────────────────────────────────────
    if modo not in ("viral", "podcast"):
        return {"error": f"Modo '{modo}' no reconocido. Modos disponibles: viral, podcast"}

    from subir_r2 import subir_clip_r2
    clips_resultado = []

    try:
        if modo == "podcast":
            import queue as _queue
            import threading as _threading

            cola      = _queue.Queue()
            error_box = [None]

            def _on_clip(local_path):
                # Llamado desde el hilo de podcast al terminar cada clip
                cola.put(local_path)

            def _run_podcast():
                try:
                    from podcast_api import procesar_podcast
                    procesar_podcast(
                        url,
                        fuente_sub    = fuente_sub,
                        mayusculas    = mayusculas,
                        modo_sub      = modo_sub,
                        on_clip_listo = _on_clip,
                    )
                except Exception as e:
                    error_box[0] = e
                    print(f"❌ Error en hilo podcast: {e}")
                    traceback.print_exc()
                finally:
                    cola.put(None)  # sentinel: podcast terminó

            hilo = _threading.Thread(target=_run_podcast, daemon=True)
            hilo.start()

            while len(clips_resultado) < MAX_CLIPS:
                try:
                    local_path = cola.get(timeout=900)  # 15 min max entre clips
                except _queue.Empty:
                    print("⚠️ Timeout: sin nuevo clip de podcast en 15 min")
                    break

                if local_path is None:  # sentinel: podcast terminó
                    break

                nombre = os.path.basename(local_path)
                try:
                    r2_url = subir_clip_r2(local_path, nombre, prefix=job_id_prefix)
                    if r2_url:
                        clip_item = {"url": r2_url, "nombre": nombre, "local": local_path}
                        clips_resultado.append(clip_item)
                        yield {"clips": [clip_item]}  # stream parcial → RunPod /stream/
                        print(f"📤 Clip {len(clips_resultado)} en stream: {r2_url}")
                except Exception as e_sub:
                    print(f"⚠️ Error subiendo clip podcast '{nombre}': {e_sub}")

            hilo.join(timeout=30)

            if error_box[0] and not clips_resultado:
                yield {"error": str(error_box[0]), "traceback": traceback.format_exc()}
                return

        else:
            # Viral: clip_listo_cb ya existe en motor_viral.py — usarlo para
            # subir + yield cada clip en tiempo real (mismo patrón que podcast)
            import queue as _queue_v
            import threading as _threading_v

            cola_viral  = _queue_v.Queue()
            error_viral = [None]

            def _on_clip_viral(local_path):
                cola_viral.put(local_path)

            def _run_viral():
                try:
                    from motor_viral import procesar_viral
                    procesar_viral(
                        url,
                        fuente_sub    = fuente_sub,
                        mayusculas    = mayusculas,
                        modo_sub      = modo_sub,
                        clip_listo_cb = _on_clip_viral,
                    )
                except Exception as e:
                    error_viral[0] = e
                    print(f"❌ Error en hilo viral: {e}")
                    traceback.print_exc()
                finally:
                    cola_viral.put(None)  # sentinel

            hilo_viral = _threading_v.Thread(target=_run_viral, daemon=True)
            hilo_viral.start()

            while len(clips_resultado) < MAX_CLIPS:
                try:
                    local_path = cola_viral.get(timeout=900)
                except _queue_v.Empty:
                    print("⚠️ Timeout: sin nuevo clip viral en 15 min")
                    break

                if local_path is None:
                    break

                nombre = os.path.basename(local_path)
                try:
                    r2_url = subir_clip_r2(local_path, nombre, prefix=job_id_prefix)
                    if r2_url:
                        clip_item = {"url": r2_url, "nombre": nombre, "local": local_path}
                        clips_resultado.append(clip_item)
                        yield {"clips": [clip_item]}
                        print(f"📤 Clip viral {len(clips_resultado)}: {r2_url}")
                except Exception as e_sub:
                    print(f"⚠️ Error subiendo clip viral '{nombre}': {e_sub}")

            hilo_viral.join(timeout=30)

            if error_viral[0] and not clips_resultado:
                yield {"error": str(error_viral[0]), "traceback": traceback.format_exc()}
                return

        print(f"✅ {len(clips_resultado)} clips totales")
        yield {
            "clips":       clips_resultado,
            "total_clips": len(clips_resultado),
        }

    except Exception as e:
        print(f"❌ Error en handler: {e}")
        traceback.print_exc()
        yield {
            "error":     str(e),
            "traceback": traceback.format_exc(),
            "clips":     clips_resultado,
        }


runpod.serverless.start({
    "handler": handler,
    # Con return_aggregate_stream=True, cuando el generator termina RunPod
    # agrega TODOS los yields como lista en el output del /status (COMPLETED).
    # Sin esto, el output queda vacío ({}) y los clips solo llegan por /stream
    # que es efímero y se puede perder. Con esto, el COMPLETED siempre
    # tiene todos los clips aunque el /stream haya fallado o timed-out.
    "return_aggregate_stream": True,
})
