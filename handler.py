"""
KLYPO — RunPod Serverless Handler

Recibe un job de RunPod, llama a procesar_viral y devuelve las rutas de los clips.

Input esperado (event["input"]):
    url        : str   — URL de YouTube o ruta local al video
    fuente_sub : str   — Anton | Arial | Montserrat | BebasNeue | Poppins  (default: Anton)
    mayusculas : bool  — True = MAYÚSCULAS en subtítulos  (default: False)
    modo_sub   : str   — bloques | karaoke | none          (default: bloques)
    modo       : str   — "viral" (único modo soportado por ahora)

Límites de costo:
    MAX_CLIPS   = 8    clips devueltos como máximo
    MAX_DUR_SEG = 1200 segundos de video (20 min) — rechaza antes de descargar
"""

import os
import sys
import subprocess
import traceback

import runpod

# modulos_virales/ está al mismo nivel que handler.py dentro del contenedor
_RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_RAIZ, "modulos_virales"))

MAX_CLIPS   = 8
MAX_DUR_SEG = 1200  # 20 minutos

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


def _duracion_video(url: str) -> float | None:
    """
    Obtiene la duración en segundos SIN descargar el video completo.
    - Para URLs de YouTube: usa yt-dlp --print duration (solo metadata).
    - Para archivos locales: usa ffprobe.
    Devuelve None si no se puede determinar (no bloquea el job).
    """
    try:
        if url.startswith("http"):
            cmd = ["yt-dlp", "--skip-download", "--print", "duration", "--no-warnings", url]
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
    fuente_sub = inp.get("fuente_sub", "Anton")
    mayusculas = bool(inp.get("mayusculas", False))
    modo_sub   = inp.get("modo_sub",   "bloques")
    # modo es para extensión futura (viral / podcast) — actualmente solo viral
    modo       = inp.get("modo",       "viral")

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
                f"Video demasiado largo: {int(dur)}s "
                f"(máximo permitido: {MAX_DUR_SEG}s = 20 min). "
                f"Proporciona un clip más corto."
            )
        }

    print(f"🚀 KLYPO handler — url={url[:60]} | fuente={fuente_sub} | modo_sub={modo_sub}")

    # ── Procesamiento ─────────────────────────────────────────────────────────
    try:
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

        print(f"✅ {len(rutas)} clips listos")
        return {
            "clips":       rutas,
            "total_clips": len(rutas),
        }

    except Exception as e:
        print(f"❌ Error en handler: {e}")
        return {
            "error":     str(e),
            "traceback": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})
