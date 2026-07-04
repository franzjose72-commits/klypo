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

    try:
        if modo == "podcast":
            from podcast_api import procesar_podcast
            rutas = procesar_podcast(
                url,
                fuente_sub = fuente_sub,
                mayusculas = mayusculas,
                modo_sub   = modo_sub,
            )
        else:
            from motor_viral import procesar_viral
            rutas = procesar_viral(
                url,
                fuente_sub = fuente_sub,
                mayusculas = mayusculas,
                modo_sub   = modo_sub,
            )

        # Limitar clips para controlar costo
        if len(rutas) > MAX_CLIPS:
            print(f"⚠️ {len(rutas)} clips generados — limitando a {MAX_CLIPS}")
            rutas = rutas[:MAX_CLIPS]

        # Subir a R2 y reemplazar rutas locales por URLs descargables
        from subir_r2 import subir_clip_r2
        clips_resultado = []
        for ruta in rutas:
            nombre = os.path.basename(ruta)
            r2_url = subir_clip_r2(ruta, nombre)
            clips_resultado.append({
                "nombre": nombre,
                "url":    r2_url or "",    # URL de R2 o vacio si fallo la subida
                "local":  ruta,            # ruta en el contenedor (referencia)
            })

        print(f"✅ {len(clips_resultado)} clips listos")
        return {
            "clips":       clips_resultado,
            "total_clips": len(clips_resultado),
        }

    except Exception as e:
        print(f"❌ Error en handler: {e}")
        return {
            "error":     str(e),
            "traceback": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})
