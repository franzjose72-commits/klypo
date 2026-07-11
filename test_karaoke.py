"""
Test rápido de subtítulos karaoke V47.
Cambia VIDEO_PATH por cualquier .mp4 que tengas en tu PC.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modulos_virales"))

import moviepy as mpy
from render_viral import render_viral

# ─── CAMBIA ESTO ────────────────────────────────────────────────────────────────
VIDEO_PATH  = r"C:\ruta\a\tu\video.mp4"   # cualquier mp4 que tengas
OUTPUT_PATH = r"C:\Users\USUARIO\Desktop\test_karaoke_out.mp4"
DURACION    = 20   # segundos del clip de prueba (para que sea rápido)
# ────────────────────────────────────────────────────────────────────────────────

# Palabras de prueba con timestamps realistas (~20s de contenido)
WORDS_TEST = [
    {"word": "este",       "start": 0.20, "end": 0.45},
    {"word": "momento",    "start": 0.50, "end": 0.95},
    {"word": "cambió",     "start": 1.00, "end": 1.45},
    {"word": "todo",       "start": 1.50, "end": 1.90},
    {"word": "en",         "start": 2.20, "end": 2.35},
    {"word": "mi",         "start": 2.40, "end": 2.55},
    {"word": "vida",       "start": 2.60, "end": 3.00},
    {"word": "porque",     "start": 3.40, "end": 3.70},
    {"word": "nadie",      "start": 3.75, "end": 4.10},
    {"word": "me",         "start": 4.15, "end": 4.30},
    {"word": "dijo",       "start": 4.35, "end": 4.70},
    {"word": "la",         "start": 4.75, "end": 4.85},
    {"word": "verdad",     "start": 4.90, "end": 5.40},
    {"word": "nunca",      "start": 6.00, "end": 6.35},
    {"word": "pensé",      "start": 6.40, "end": 6.80},
    {"word": "que",        "start": 6.85, "end": 6.95},
    {"word": "llegaría",   "start": 7.00, "end": 7.55},
    {"word": "tan",        "start": 7.60, "end": 7.75},
    {"word": "lejos",      "start": 7.80, "end": 8.30},
    {"word": "pero",       "start": 8.80, "end": 9.05},
    {"word": "aquí",       "start": 9.10, "end": 9.45},
    {"word": "estoy",      "start": 9.50, "end": 9.90},
    {"word": "contándote", "start": 9.95, "end": 10.60},
    {"word": "lo",         "start": 10.65, "end": 10.75},
    {"word": "que",        "start": 10.80, "end": 10.90},
    {"word": "pasó",       "start": 10.95, "end": 11.40},
    {"word": "y",          "start": 12.00, "end": 12.10},
    {"word": "fue",        "start": 12.15, "end": 12.45},
    {"word": "increíble",  "start": 12.50, "end": 13.10},
    {"word": "de",         "start": 13.50, "end": 13.60},
    {"word": "verdad",     "start": 13.65, "end": 14.10},
    {"word": "no",         "start": 14.15, "end": 14.30},
    {"word": "lo",         "start": 14.35, "end": 14.45},
    {"word": "podía",      "start": 14.50, "end": 14.85},
    {"word": "creer",      "start": 14.90, "end": 15.40},
]

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ No encontré el video: {VIDEO_PATH}")
        print("   Cambia VIDEO_PATH en la línea 10 de este script.")
        return

    print(f"📂 Cargando video: {VIDEO_PATH}")
    clip = mpy.VideoFileClip(VIDEO_PATH).subclipped(0, min(DURACION, mpy.VideoFileClip(VIDEO_PATH).duration))

    print(f"🎬 Renderizando con karaoke V47 ({len(WORDS_TEST)} palabras de prueba)...")
    render_viral(
        clip,
        OUTPUT_PATH,
        words=WORDS_TEST,
        modo_sub="karaoke",
        mayusculas=True,   # prueba en mayúsculas como en producción
    )

    clip.close()

    if os.path.exists(OUTPUT_PATH):
        print(f"\n✅ Listo: {OUTPUT_PATH}")
        os.startfile(OUTPUT_PATH)   # abre el video directamente en Windows
    else:
        print("❌ No se generó el archivo de salida.")

if __name__ == "__main__":
    main()
