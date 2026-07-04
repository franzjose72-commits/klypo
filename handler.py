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
    MAX_CLIPS   = 15   clips devueltos como maximo
    MAX_DUR_SEG = 7200 segundos de video (2 horas) — rechaza antes de descargar
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

MAX_CLIPS   = 15
MAX_DUR_SEG = 7200  # 2 horas

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

    # ── Validación de entrada ─────────────────────────────────────────────────
    if not url:
        return {"error": "Falta el campo 'url' en el input"}

    # ── Descarga directa para videos subidos por el usuario (no YouTube) ──────
    # Si la URL es de R2 u otro CDN (no YouTube), descargamos el archivo localmente
    # antes de pasarlo a procesar_viral / procesar_podcast.
    # Ambos modos aceptan rutas locales, por lo que el resto del pipeline no cambia.
    if url.startswith("http") and not _es_youtube(url):
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

    # ── Límite de duración (antes de descargar) ───────────────────────────────
    dur = _duracion_video(url)
    if dur is not None and dur > MAX_DUR_SEG:
        return {
            "error": (
                f"Video demasiado largo: {int(dur // 60):.0f} min "
                f"(maximo permitido: {MAX_DUR_SEG // 3600:.0f} horas = {MAX_DUR_SEG} s)."
            )
        }

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
                    r2_url = subir_clip_r2(local_path, nombre)
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
            # Viral: motor_viral.py NO se toca — devuelve todas las rutas juntas
            from motor_viral import procesar_viral
            rutas = procesar_viral(
                url,
                fuente_sub = fuente_sub,
                mayusculas = mayusculas,
                modo_sub   = modo_sub,
            )

            if len(rutas) > MAX_CLIPS:
                print(f"⚠️ {len(rutas)} clips generados — limitando a {MAX_CLIPS}")
                rutas = rutas[:MAX_CLIPS]

            for ruta in rutas:
                nombre = os.path.basename(ruta)
                try:
                    r2_url = subir_clip_r2(ruta, nombre)
                    if r2_url:
                        clip_item = {"url": r2_url, "nombre": nombre, "local": ruta}
                        clips_resultado.append(clip_item)
                        yield {"clips": [clip_item]}  # un yield por clip viral
                        print(f"📤 Clip viral {len(clips_resultado)}/{len(rutas)}: {r2_url}")
                except Exception as e_sub:
                    print(f"⚠️ Error subiendo clip viral '{nombre}': {e_sub}")

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
