"""
KLYPO VIRAL — Motor Principal V10 (Clipping)

FLUJO:
  1. Descarga o carga el video
  2. Transcribe audio con Whisper (segmento + palabra para subtítulos)
  3. Llama elige 2 clips virales por TIPO (REVELACION/HUMOR/EMOCIONAL/DATO)
     - REVELACION: hook primero aunque sea del minuto 40, luego contexto
     - HUMOR: empieza justo antes del chiste, termina con la reacción
     - EMOCIONAL: historia personal, vulnerabilidad, fracaso→superación
     - DATO: afirmación contraintuitiva primero, luego la explicación
  4. Arma cada clip (segmentos en el orden que Llama decide)
  5. Renderiza: letterbox 9:16 + subtítulos Impact quemados

FALLBACK (sin habla): picos de audio + movimiento.

INDEPENDENCIA TOTAL: no toca camara.py, editor.py ni transcriptor.py.
"""

import os
import sys
import unicodedata
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import moviepy as mpy

from detector_viral import (
    detectar_picos_audio,
    detectar_picos_movimiento,
    fusionar_en_clips,
    transcribir_video_completo,
    detectar_clips_ia,
)
from render_viral import render_viral
from broll_viral import procesar_broll

VERSION = "V80"
CARPETA = f"CLIPS_VIRAL_V80"
DUR_MIN = 11
DUR_MAX = 110


def _sanitizar(texto):
    nfkd = unicodedata.normalize("NFD", texto)
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    return "".join(c for c in s if c.isascii() and (c.isalnum() or c in " _-")).strip()[:45]


def _obtener_video(fuente):
    fuente = fuente.strip()
    if not fuente.startswith("http"):
        if not os.path.exists(fuente):
            print(f"❌ Archivo no encontrado: {fuente}")
            return None
        return fuente

    print("⬇️  Descargando de YouTube...")
    import subprocess, glob

    raiz    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cookies = os.path.join(raiz, "llave.txt")
    output  = os.path.join(raiz, "viral_download.%(ext)s")

    for viejo in glob.glob(os.path.join(raiz, "viral_download.*")):
        try: os.remove(viejo)
        except: pass

    proxy_user = os.environ.get("PROXY_USER", "").strip()
    proxy_pass = os.environ.get("PROXY_PASS", "").strip()
    proxy_args = ["--proxy", f"http://{proxy_user}:{proxy_pass}@gw.dataimpulse.com:823"] \
                 if proxy_user and proxy_pass else []
    if proxy_args:
        print("🌐 Usando proxy residencial DataImpulse")

    cmd = [
        "yt-dlp", "--cookies", cookies,
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        "-o", output, "--no-playlist",
        "--extractor-args", "youtube:player_client=android,web",
        "--verbose",
        *proxy_args,
        fuente,
    ]
    try:
        subprocess.run(cmd, check=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        archivos = glob.glob(os.path.join(raiz, "viral_download.*"))
        return archivos[0] if archivos else None
    except subprocess.CalledProcessError as e:
        salida = (e.output or "").strip()
        raise ValueError(f"No se pudo obtener el video. yt-dlp output:\n{salida[-1500:] if salida else '(vacio)'}")
    except Exception as e:
        raise ValueError(f"No se pudo obtener el video: {e}")



def _remap_words(words_global, segmentos_clip):
    """
    Remapea timestamps de palabras del video original al timeline del clip ensamblado.
    Los segmentos se procesan EN EL ORDEN DADO (pueden ser no cronológicos → hook-first).
    """
    clip_words   = []
    clip_offset  = 0.0

    for seg in segmentos_clip:
        seg_ini = float(seg["inicio"])
        seg_fin = float(seg["fin"])
        seg_dur = seg_fin - seg_ini

        for w in words_global:
            w_s = float(w["start"])
            w_e = float(w["end"])
            if w_s >= seg_ini - 0.05 and w_e <= seg_fin + 0.1:
                clip_words.append({
                    "word":    w["word"],
                    "start":   clip_offset + max(0.0, w_s - seg_ini),
                    "end":     clip_offset + min(w_e - seg_ini, seg_dur),
                    "new_seg": w.get("new_seg", False),
                })

        clip_offset += seg_dur

    return clip_words


def _armar_clip(video, segmentos, duracion_total):
    """Corta y concatena los segmentos EN EL ORDEN DADO. Retorna VideoClip o None."""
    partes = []
    for seg in segmentos:
        ini = max(0.0, float(seg["inicio"]))
        fin = min(duracion_total, float(seg["fin"]))
        if fin - ini < 2.0:
            print(f"   ⚠️ Segmento {ini:.0f}s-{fin:.0f}s muy corto, saltado")
            continue
        try:
            partes.append(video.subclipped(ini, fin))
        except Exception as e:
            print(f"   ⚠️ Error cortando {ini:.0f}s-{fin:.0f}s: {e}")
    if not partes:
        return None
    return mpy.concatenate_videoclips(partes) if len(partes) > 1 else partes[0]


def procesar_viral(url, fuente_sub="Arial", mayusculas=False, modo_sub="bloques", progreso_cb=None, clip_listo_cb=None):
    """
    Núcleo de procesamiento — sin input(), sin menus.
    Recibe parámetros directamente y devuelve lista de rutas absolutas de los clips.
    Llamada tanto desde ejecutar_viral() (terminal) como desde api.py (web).
    """
    try:
        video_path = _obtener_video(url)
    except ValueError:
        raise  # ya trae el detalle de yt-dlp — propaga sin modificar
    if not video_path:
        raise ValueError("No se pudo obtener el video: yt-dlp no genero archivo de salida (el video puede requerir autenticacion o estar restringido)")

    video    = mpy.VideoFileClip(video_path)
    duracion = video.duration
    print(f"   📹 Duración: {duracion:.0f}s ({duracion / 60:.1f} min)\n")

    os.makedirs(CARPETA, exist_ok=True)
    clips_detectados = []
    words_global     = []

    # ── FASE 1: Transcripción ─────────────────────────────────────────────────
    print("🎙️ FASE 1: Transcribiendo...")
    tx           = transcribir_video_completo(video_path)
    transcript   = tx.get("text",  "") if isinstance(tx, dict) else tx
    words_global = tx.get("words", []) if isinstance(tx, dict) else []

    if transcript.strip():
        print(f"   ✅ {len(transcript.split())} palabras | {len(words_global)} timestamps\n")
        print("🧠 FASE 2: Llama detectando clips...")
        clips_detectados = detectar_clips_ia(transcript, duracion)

        if clips_detectados:
            print(f"\n   ✅ {len(clips_detectados)} clips detectados:")
            for i, c in enumerate(clips_detectados):
                dur_t  = sum(float(s["fin"]) - float(s["inicio"]) for s in c["segmentos"])
                segs_s = " → ".join(f"{float(s['inicio']):.0f}s-{float(s['fin']):.0f}s" for s in c["segmentos"])
                print(f"   [{i+1}] [{c.get('tipo','?')}] '{c['titulo']}' — {segs_s} ({dur_t:.0f}s)")

            clips_validos = []
            for c in clips_detectados:
                dur_clip = sum(float(s["fin"]) - float(s["inicio"]) for s in c["segmentos"])
                if dur_clip < DUR_MIN:
                    print(f"   ⏭️ Descartado '{c['titulo']}' — {dur_clip:.0f}s < {DUR_MIN}s")
                elif dur_clip > DUR_MAX:
                    print(f"   ⏭️ Descartado '{c['titulo']}' — {dur_clip:.0f}s > {DUR_MAX}s")
                else:
                    clips_validos.append(c)
            clips_detectados = clips_validos

            if not clips_detectados:
                print("   ⚠️ Sin clips válidos → fallback")
        else:
            print("   ⚠️ Llama no devolvió clips → fallback")
    else:
        print("   ⚠️ Sin transcripción → fallback\n")

    # ── Fallback: picos audio + movimiento ────────────────────────────────────
    if not clips_detectados:
        print("🔊 FALLBACK: Detectando por picos...")
        picos_audio = detectar_picos_audio(video_path)
        picos_video = detectar_picos_movimiento(video_path)
        segs = fusionar_en_clips(
            picos_audio, picos_video, [],
            duracion_total=duracion, dur_min=DUR_MIN, dur_max=DUR_MAX,
        )
        for seg in sorted(segs, key=lambda x: x["intensidad"], reverse=True):
            clips_detectados.append({
                "tipo":      "PICO",
                "titulo":    f"momento_t{int(seg['t_pico'])}s",
                "razon":     f"Pico {seg['tipo']} score {seg['intensidad']:.1f}",
                "segmentos": [{"inicio": seg["inicio"], "fin": seg["fin"]}],
            })
        if not clips_detectados:
            video.close()
            raise ValueError("No se encontraron momentos virales en el video")

    # ── Renderizar ────────────────────────────────────────────────────────────
    n = len(clips_detectados)
    msg_render = f"✅ Encontré {n} momento{'s' if n != 1 else ''} viral{'es' if n != 1 else ''}, generando clips..."
    print(f"\n🎬 {msg_render}\n")
    if progreso_cb:
        progreso_cb(msg_render)
    rutas_generadas = []

    for i, info in enumerate(clips_detectados):
        tipo   = info.get("tipo", "?")
        titulo = info["titulo"]
        print(f"   ── Clip {i+1} [{tipo}]: '{titulo}'")
        try:
            clip = _armar_clip(video, info["segmentos"], duracion)
            if clip is None:
                print(f"   ⚠️ No se pudo armar el clip.\n")
                continue

            words_clip = _remap_words(words_global, info["segmentos"]) if words_global else []
            print(f"   ⏱️  {clip.duration:.0f}s | {len(info['segmentos'])} seg | {len(words_clip)} palabras")

            nombre = _sanitizar(titulo) or f"clip_{i+1}"
            output = os.path.join(CARPETA, f"KLYPO_{VERSION}_{i+1:02d}_{tipo}_{nombre}.mp4")

            render_viral(
                clip, output,
                words=words_clip if words_clip else None,
                titulo=info.get("titulo"),
                fuente_sub=fuente_sub,
                modo_sub=modo_sub,
                mayusculas=mayusculas,
            )

            if words_clip and os.path.exists(output):
                procesar_broll(words_clip, clip.duration, output, fuente_broll="local")

            ruta_abs = os.path.abspath(output)
            rutas_generadas.append(ruta_abs)
            print(f"   ✅ {output}\n")
            if clip_listo_cb:
                clip_listo_cb(ruta_abs)

        except Exception as e:
            print(f"   ⚠️ Error clip {i+1}: {e}")
            traceback.print_exc()

    video.close()
    print(f"🏁 KLYPO VIRAL {VERSION} → {len(rutas_generadas)}/{len(clips_detectados)} clips en '{CARPETA}/'")
    return rutas_generadas


def ejecutar_viral():
    """Wrapper para uso por terminal — mantiene los menus interactivos."""
    print(f"⚡ KLYPO VIRAL {VERSION} — Clipping IA")
    print(f"   Whisper → Llama detecta clips virales | Zoom auto | B-roll | 9:16")
    print("─" * 60)

    fuente = input("🎬 Ruta del video o URL de YouTube: ")

    print("\n¿Qué fuente de subtítulos?")
    print("  1. Arial       (fina, estilo caption natural — recomendada)")
    print("  2. Anton       (impacto máximo)")
    print("  3. Montserrat  (Black)")
    print("  4. Bebas Neue")
    print("  5. Poppins     (Black)")
    _fs = input("Elige [1-5] (Enter = Arial): ").strip()
    fuente_sub = {"1": "Arial", "2": "Anton", "3": "Montserrat", "4": "BebasNeue", "5": "Poppins"}.get(_fs, "Arial")

    print("\n¿Mayúsculas o minúsculas?")
    print("  1. minúsculas  (natural — recomendado)")
    print("  2. MAYÚSCULAS  (impacto)")
    _mc = input("Elige [1/2] (Enter = minúsculas): ").strip()
    mayusculas = (_mc == "2")

    print("\n¿Qué estilo de subtítulos?")
    print("  1. Bloques  (3-4 palabras, natural — recomendado)")
    print("  2. Karaoke  (palabras se iluminan al hablarse)")
    print("  3. Sin subtítulos")
    _ms = input("Elige [1/2/3] (Enter = Bloques): ").strip()
    modo_sub = {"2": "karaoke", "3": "none"}.get(_ms, "bloques")
    print()

    try:
        clips = procesar_viral(fuente, fuente_sub, mayusculas, modo_sub)
        if clips:
            print(f"\n✅ {len(clips)} clips generados:")
            for c in clips:
                print(f"   {c}")
    except ValueError as e:
        print(f"❌ {e}")


if __name__ == "__main__":
    ejecutar_viral()
