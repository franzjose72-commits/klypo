"""
KLYPO — Modo Podcast en RunPod (API no interactiva)

Replica la logica de editor.ejecutar_nero() sin llamadas a input().
Descarga con las estrategias android/tv/ios/web+bgutil y proxy DataImpulse
definidas directamente aqui (sin importar motor_viral ni sus modulos de IA).

editor.py y ejecutar_nero() quedan intactos para uso en terminal.
"""

import os
import sys
import glob
import json
import re
import subprocess
import unicodedata
import cv2

_RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_RAIZ, "modulos_virales"))

import moviepy as mpy

# Fuentes disponibles en Linux (fonts/ en el contenedor)
_FUENTES_LINUX = {"montserrat", "poppins", "anton", "bebas"}

# Mapeo desde valores de API (capitalizados) a internos (minusculas de subtitulos.py)
_FUENTE_API = {
    "Anton":      "anton",
    "Montserrat": "montserrat",
    "BebasNeue":  "bebas",
    "Poppins":    "poppins",    # Poppins.ttf esta en fonts/ — usar directamente
    "Arial":      "anton",      # Arial no existe en Linux → fallback Anton
    "impact":     "anton",      # por si llega en minusculas
}


def _fuente_segura(fuente_sub_api: str) -> str:
    """Devuelve la fuente interna garantizando que el archivo exista en Linux."""
    resultado = _FUENTE_API.get(fuente_sub_api, "anton")
    return resultado if resultado in _FUENTES_LINUX else "anton"


def _extraer_json(texto: str) -> list:
    match = re.search(r'(\[.*?\]|\{.*?\})', texto, re.DOTALL)
    if match:
        try:
            r = json.loads(match.group(1))
            return [r] if isinstance(r, dict) else r
        except Exception:
            pass
    return []


def _sanitizar_titulo(titulo: str) -> str:
    nfkd = unicodedata.normalize('NFD', titulo)
    sin_acentos = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return ''.join(
        c for c in sin_acentos if c.isascii() and (c.isalnum() or c in ' _-')
    ).strip()[:50]


def _descargar_podcast(url: str) -> str:
    """
    Descarga el video con las mismas 4 estrategias del modo viral:
    android (sin cookies) → tv (con cookies) → ios → web+bgutil.
    Definida aqui para no importar motor_viral (evita cargar modulos de IA).
    """
    url = url.strip()
    if not url.startswith("http"):
        if os.path.exists(url):
            return url
        raise ValueError(f"Archivo local no encontrado: {url}")

    print("⬇️  Descargando podcast de YouTube...")

    cookies    = os.path.join(_RAIZ, "llave.txt")
    output_tpl = os.path.join(_RAIZ, "podcast_download.%(ext)s")

    proxy_user = os.environ.get("PROXY_USER", "").strip()
    proxy_pass = os.environ.get("PROXY_PASS", "").strip()
    proxy_args = []
    if proxy_user and proxy_pass:
        proxy_args = ["--proxy", f"http://{proxy_user}:{proxy_pass}@gw.dataimpulse.com:823"]
        print("🌐 Proxy DataImpulse activo")
    else:
        print("⚠️  Sin proxy — descarga directa")

    bgutil_args = [
        "--extractor-args",
        "youtubepot-bgutilscript:server_home=/root/bgutil-ytdlp-pot-provider/server",
    ]

    estrategias = [
        {
            "nombre":  "tv (con cookies, hasta 1080p)",
            "client":  "tv",
            "formato": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b",
            "cookies": True,
        },
        {
            "nombre":  "web+bgutil (con cookies, hasta 1080p)",
            "client":  "web",
            "formato": "bv*[height<=1080]+ba/b",
            "cookies": True,
        },
        {
            "nombre":  "ios (sin cookies)",
            "client":  "ios",
            "formato": "best[height<=1080][ext=mp4]/best[ext=mp4]/best",
            "cookies": False,
        },
        {
            "nombre":  "android (sin cookies, 480p fallback)",
            "client":  "android",
            "formato": "18/best[ext=mp4]/best",
            "cookies": False,
        },
    ]

    errores = []
    for i, est in enumerate(estrategias):
        print(f"🎯 Intento {i+1}/{len(estrategias)}: {est['nombre']}")

        for viejo in glob.glob(os.path.join(_RAIZ, "podcast_download.*")):
            try:
                os.remove(viejo)
            except Exception:
                pass

        cmd = ["yt-dlp"]
        if est["cookies"] and os.path.exists(cookies):
            cmd += ["--cookies", cookies]
        cmd += [
            "-f", est["formato"],
            "--merge-output-format", "mp4",
            "-o", output_tpl,
            "--no-playlist",
            "--extractor-args", f"youtube:player_client={est['client']}",
        ]
        if est["client"] == "web":
            cmd += bgutil_args
        cmd += ["--verbose"] + proxy_args + [url]

        try:
            subprocess.run(
                cmd, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            archivos = glob.glob(os.path.join(_RAIZ, "podcast_download.*"))
            if archivos:
                print(f"✅ Descarga OK: {est['nombre']}")
                return archivos[0]
            errores.append(f"[{est['nombre']}] archivo no encontrado tras descarga")
        except subprocess.CalledProcessError as e:
            salida  = (e.output or "").strip()
            resumen = salida[-400:] if salida else "(sin output)"
            print(f"❌ Estrategia {est['nombre']} fallida")
            errores.append(f"[{est['nombre']}] {resumen}")

    raise ValueError(
        f"No se pudo descargar el podcast tras {len(estrategias)} estrategias.\n"
        + "\n---\n".join(errores[-2:])
    )


def procesar_podcast(
    url: str,
    fuente_sub: str = "Anton",
    mayusculas: bool = False,
    modo_sub: str = "karaoke",
    on_clip_listo = None,
) -> list:
    """
    Version no interactiva de editor.ejecutar_nero() para RunPod.

    Parametros:
        url        — URL de YouTube o ruta local al video
        fuente_sub — Anton | Montserrat | BebasNeue | Arial | Poppins
                     (Arial/Poppins/Impact sin archivo en Linux → Anton)
        mayusculas — aceptado por compatibilidad; subtitulos.py aplica UPPER internamente
        modo_sub   — karaoke | bloques | none

    Devuelve:
        Lista de rutas absolutas a los clips generados (.mp4)
    """
    import time as _time
    _t0 = _time.time()

    print(f"⏱️ [+{_time.time()-_t0:.1f}s] Importando transcriptor...")
    from transcriptor import transcribir_clip_timestamps, procesar_segmentos_paralelo
    print(f"⏱️ [+{_time.time()-_t0:.1f}s] transcriptor OK")

    print(f"⏱️ [+{_time.time()-_t0:.1f}s] Importando camara...")
    from camara import nero_reframe
    print(f"⏱️ [+{_time.time()-_t0:.1f}s] camara OK")

    print(f"⏱️ [+{_time.time()-_t0:.1f}s] Importando subtitulos...")
    from subtitulos import agregar_subtitulos
    print(f"⏱️ [+{_time.time()-_t0:.1f}s] subtitulos OK")

    fuente_interna = _fuente_segura(fuente_sub)
    con_subs = (modo_sub != "none")

    print(f"🎙️ KLYPO Podcast — fuente={fuente_interna} | modo_sub={modo_sub} | con_subs={con_subs}")

    # ── 1. Descarga ──────────────────────────────────────────────────────────────
    video_path = _descargar_podcast(url)

    video = mpy.VideoFileClip(video_path)
    duracion_total = video.duration
    print(f"📹 Video: {duracion_total:.0f}s ({duracion_total / 60:.1f} min)")

    # ── 2. Extraccion de audio por chunks de 10 min ───────────────────────────────
    # Comienza en el segundo 150 (min 2:30) para saltar la intro editada
    print("🧠 Analizando desde el min 2:30 (chunks de 10 min)...")
    segmentos_audio = []
    for start in range(150, int(duracion_total), 600):
        fin_segmento = min(start + 600, duracion_total)
        temp_audio = f"temp_podcast_{start}.mp3"
        try:
            video.audio.subclipped(start, fin_segmento).write_audiofile(
                temp_audio, bitrate="16k", logger=None
            )
            segmentos_audio.append((start, temp_audio))
        except Exception as e:
            print(f"⚠️ Error extrayendo audio {start}s: {e}")

    # max_workers=1: secuencial para no saturar el rate limit de Groq
    resultados = procesar_segmentos_paralelo(segmentos_audio, duracion_total, max_workers=1)

    guion_ganador = []
    for start, ganchos_txt in resultados:
        clips = _extraer_json(ganchos_txt)
        for clip in clips:
            try:
                clip['inicio'] = float(clip['inicio']) + start
                clip['fin']    = float(clip['fin']) + start
                if clip['inicio'] < 150:
                    clip['inicio'] = 150
                if clip['inicio'] >= duracion_total or clip['fin'] > duracion_total:
                    continue
                guion_ganador.append(clip)
            except Exception:
                continue

    for _, temp_audio in segmentos_audio:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

    if not guion_ganador:
        video.close()
        raise ValueError("KLYPO no encontro clips virales en el video")

    # ── 3. Deduplicacion (>50 % de overlap) ──────────────────────────────────────
    guion_limpio = []
    for clip in guion_ganador:
        es_dup = False
        for existente in guion_limpio:
            ini_ov  = max(clip['inicio'], existente['inicio'])
            fin_ov  = min(clip['fin'],   existente['fin'])
            overlap = fin_ov - ini_ov
            dur_c   = clip['fin'] - clip['inicio']
            if overlap > dur_c * 0.5:
                es_dup = True
                break
        if not es_dup:
            guion_limpio.append(clip)

    guion_ganador = guion_limpio
    print(f"✅ {len(guion_ganador)} clips unicos encontrados.")

    # ── 4. Edicion y renderizado ──────────────────────────────────────────────────
    folder = "CLIPS_PODCAST_KLYPO"
    os.makedirs(folder, exist_ok=True)
    rutas_generadas = []

    for i, info in enumerate(guion_ganador):
        try:
            inicio   = float(info['inicio'])
            fin      = float(info['fin'])
            duracion = fin - inicio

            if duracion < 30 or duracion > 110:
                print(f"⏭️ Clip {i+1} saltado ({duracion:.0f}s fuera de rango 30-110s)")
                continue

            inicio = max(150, inicio)
            fin    = min(fin, duracion_total - 1)

            if inicio >= duracion_total or fin <= inicio:
                print(f"⏭️ Clip {i+1} saltado (inicio {inicio:.0f}s fuera del video)")
                continue

            titulo         = info.get('titulo', 'Clip_Podcast')
            titulo_archivo = _sanitizar_titulo(titulo)
            print(f"✨ Clip {i+1}: '{titulo}' ({duracion:.0f}s)")

            clip = video.subclipped(inicio, fin)

            # Transcribir con timestamps para subtitulos
            words_sub      = []
            audio_sub_path = f"temp_sub_podcast_{i}.wav"
            try:
                if con_subs and clip.audio:
                    clip.audio.write_audiofile(
                        audio_sub_path, fps=16000, nbytes=2,
                        ffmpeg_params=["-ac", "1"], logger=None
                    )
                    words_sub = transcribir_clip_timestamps(audio_sub_path)
                    if words_sub:
                        print(f"   📝 {len(words_sub)} palabras para subtitulos")
                    else:
                        print("   ⚠️ Whisper devolvio 0 palabras")
            except Exception as e:
                import traceback as _tb
                print(f"   ❌ Error subtitulos: {e}")
                _tb.print_exc()
            finally:
                if os.path.exists(audio_sub_path):
                    os.remove(audio_sub_path)

            # Auto-trim: recortar silencio muerto al final si el LLM fue generoso
            if words_sub:
                fin_palabras = inicio + words_sub[-1]["end"] + 0.5
                if fin_palabras < fin - 2.0:
                    ahorro = fin - fin_palabras
                    fin    = fin_palabras
                    clip   = clip.subclipped(0, fin - inicio)
                    print(f"   ✂️ Trim -{ahorro:.1f}s de silencio")

            # Reencuadre 9:16 con deteccion de caras y VAD (nero_reframe abre su propio lector)
            final = nero_reframe(video_path, clip, inicio, fin)

            # nero_reframe recorta los frames a 9:16 (p.ej. 607×1080 para fuente 1920×1080)
            # pero clip.size queda en las dimensiones originales (1920×1080).
            # FFMPEG recibe metadata incorrecta → coloca el recorte en una caja más grande
            # con barras negras por todos los lados.
            # Fix: escalar a 1080×1920 estándar y corregir el metadata del clip.
            def _escalar_1080x1920(get_frame, t):
                f = get_frame(t)
                h, w = f.shape[:2]
                if w == 1080 and h == 1920:
                    return f
                return cv2.resize(f, (1080, 1920), interpolation=cv2.INTER_LINEAR)
            final = final.transform(_escalar_1080x1920)
            final.size = (1080, 1920)

            if words_sub and con_subs:
                final = agregar_subtitulos(final, words_sub, fuente=fuente_interna, modo=modo_sub)

            # Fade in/out suave 0.5s estilo CapCut
            _dur = final.duration
            def _fades(get_frame, t):
                frame = get_frame(t).astype('float32')
                if t < 0.5:
                    frame *= t / 0.5
                elif t > _dur - 0.5:
                    frame *= max(0.0, (_dur - t) / 0.5)
                return frame.astype('uint8')
            final = final.transform(_fades)

            output_path = os.path.join(folder, f"KLYPO_PODCAST_{i+1:02d}_{titulo_archivo}.mp4")
            final.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=24,
                ffmpeg_params=["-preset", "ultrafast"],
                threads=4,
                logger=None,
            )
            # NO cerrar clip ni final aqui: comparten el reader FFMPEG de video (el padre).
            # Cerrar un subclip dentro del loop mata el reader compartido y rompe los
            # siguientes clips con "'NoneType' has no attribute 'stdout'".
            # video.close() se llama UNA sola vez al salir del loop (ver abajo).
            rutas_generadas.append(os.path.abspath(output_path))
            if on_clip_listo:
                try:
                    on_clip_listo(os.path.abspath(output_path))
                except Exception as _cb_e:
                    print(f"   ⚠️ Callback on_clip_listo: {_cb_e}")

        except Exception as e:
            import traceback as _tb
            print(f"⚠️ Error editando clip {i+1}: {e}")
            _tb.print_exc()
            continue

    video.close()
    print(f"🏁 {len(rutas_generadas)} clips podcast generados en '{folder}'.")
    return rutas_generadas
