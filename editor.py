import json, os, unicodedata
import moviepy as mpy
from descargador import descargar_video_youtube
from transcriptor import transcribir_segmento, transcribir_clip_timestamps, buscar_ganchos_en_segmento, procesar_segmentos_paralelo
from camara import nero_reframe
from subtitulos import agregar_subtitulos

def extraer_json_de_texto(texto):
    import re
    match = re.search(r'(\[.*?\]|\{.*?\})', texto, re.DOTALL)
    if match:
        try:
            resultado = json.loads(match.group(1))
            if isinstance(resultado, dict):
                resultado = [resultado]
            return resultado
        except:
            pass
    return []

def ejecutar_nero():
    print("🚀 KLYPO.IA V89: Desde min 2:30, foco en quien habla, trim silencio.")
    link = input("🔗 Link del podcast: ")
    video_path = descargar_video_youtube(link)
    if not video_path:
        print("❌ No se pudo descargar. Abortando.")
        return

    video = mpy.VideoFileClip(video_path)
    duracion_total = video.duration

    print("\n¿Qué fuente de subtítulos?")
    print("  1. Montserrat Black  (recomendada — limpia y potente)")
    print("  2. Anton             (impacto máximo, estilo TikTok)")
    print("  3. Bebas Neue        (elegante, estilo poster)")
    print("  4. Impact            (clásico viral)")
    print("  5. Sin subtítulos")
    _sf = input("Elige [1-5] (Enter = Montserrat): ").strip()
    _fuente_map = {"1": "montserrat", "2": "anton", "3": "bebas", "4": "impact"}
    fuente_sub  = _fuente_map.get(_sf, "montserrat")
    con_subs    = (_sf != "5")

    modo_sub = "karaoke"
    if con_subs:
        print("¿Qué estilo de subtítulos?")
        print("  1. Karaoke  (palabra activa en amarillo — recomendado)")
        print("  2. Bloques  (3-4 palabras a la vez, todas blancas)")
        _sm = input("Elige [1/2] (Enter = Karaoke): ").strip()
        modo_sub = "bloques" if _sm == "2" else "karaoke"
    print()

    guion_ganador = []

    print("🧠 KLYPO analizando desde el minuto 2:30 (chunks de 10 min)...")
    # Chunks de 10 min: la mitad de llamadas API → menos rate limits, mismo contexto útil
    segmentos_audio = []
    for start in range(150, int(duracion_total), 600):
        fin_segmento = min(start + 600, duracion_total)
        temp_audio = f"temp_{start}.mp3"
        try:
            video.audio.subclipped(start, fin_segmento).write_audiofile(
                temp_audio, bitrate="16k", logger=None
            )
            segmentos_audio.append((start, temp_audio))
        except Exception as e:
            print(f"⚠️ Error extrayendo audio {start}s: {e}")

    # Paso 2: transcribir + detectar ganchos de forma secuencial (1 a la vez).
    # max_workers=3 causaba tormenta sincronizada: todos golpeaban el rate limit al mismo tiempo,
    # esperaban 60s juntos, y volvían a fallar juntos → 0 clips. Con 1 worker, el quota
    # de Groq se recupera entre segmentos y los clips salen.
    resultados = procesar_segmentos_paralelo(segmentos_audio, duracion_total, max_workers=1)

    # Paso 3: procesar resultados y limpiar archivos temporales
    for start, ganchos_txt in resultados:
        clips = extraer_json_de_texto(ganchos_txt)
        for clip in clips:
            try:
                clip['inicio'] = float(clip['inicio']) + start
                clip['fin']    = float(clip['fin']) + start
                if clip['inicio'] < 150:
                    clip['inicio'] = 150
                # Descartar clips que el LLM inventó más allá de la duración real del video
                if clip['inicio'] >= duracion_total or clip['fin'] > duracion_total:
                    continue
                guion_ganador.append(clip)
            except:
                continue
    for start, temp_audio in segmentos_audio:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

    if not guion_ganador:
        print("😔 KLYPO no encontró clips virales.")
        video.close()
        return

    guion_limpio = []
    for clip in guion_ganador:
        es_duplicado = False
        for existente in guion_limpio:
            inicio_overlap = max(clip['inicio'], existente['inicio'])
            fin_overlap = min(clip['fin'], existente['fin'])
            overlap = fin_overlap - inicio_overlap
            duracion_clip = clip['fin'] - clip['inicio']
            if overlap > duracion_clip * 0.5:
                es_duplicado = True
                break
        if not es_duplicado:
            guion_limpio.append(clip)

    guion_ganador = guion_limpio
    print(f"\n✅ {len(guion_ganador)} clips únicos encontrados.")
    print(f"🎬 Editando...\n")

    folder = "CLIPS_KLYPO_V89"
    os.makedirs(folder, exist_ok=True)
    clips_editados = 0

    for i, info in enumerate(guion_ganador):
        try:
            inicio = float(info['inicio'])
            fin = float(info['fin'])
            duracion = fin - inicio

            if duracion < 30 or duracion > 110:
                print(f"⏭️ Clip {i+1} saltado ({duracion:.0f}s fuera de rango)")
                continue

            inicio = max(150, inicio)
            fin = min(fin, duracion_total - 1)

            # Guardia de seguridad: inicio fuera del video tras clampeo
            if inicio >= duracion_total or fin <= inicio:
                print(f"⏭️ Clip {i+1} saltado (inicio {inicio:.0f}s >= duracion {duracion_total:.0f}s)")
                continue

            titulo = info.get('titulo', 'Clip_Viral')
            # Quitar acentos y caracteres no-ASCII para evitar error FFMPEG en Windows
            _nfkd = unicodedata.normalize('NFD', titulo)
            _sin_acentos = ''.join(c for c in _nfkd if not unicodedata.combining(c))
            titulo_archivo = "".join(c for c in _sin_acentos if c.isascii() and (c.isalnum() or c in (' ', '_', '-'))).strip()[:50]

            print(f"✨ Editando Clip {clips_editados+1}: '{titulo}' ({duracion:.0f}s)")

            clip = video.subclipped(inicio, fin)

            # Subtítulos: extraer audio del clip y transcribir con timestamps de Whisper
            words_sub = []
            audio_sub_path = f"temp_sub_{i}.wav"
            try:
                if clip.audio:
                    clip.audio.write_audiofile(
                        audio_sub_path, fps=16000, nbytes=2,
                        ffmpeg_params=["-ac", "1"], logger=None
                    )
                    words_sub = transcribir_clip_timestamps(audio_sub_path)
                    print(f"   📝 {len(words_sub)} palabras para subtítulos" if words_sub else "   ⚠️ Whisper devolvió 0 palabras")
            except Exception as e:
                import traceback
                print(f"   ❌ Error en subtítulos: {e}")
                traceback.print_exc()
            finally:
                if os.path.exists(audio_sub_path):
                    os.remove(audio_sub_path)

            # Auto-trim: si la última palabra transcrita termina mucho antes del fin
            # marcado por Llama, recortar el silencio muerto para un corte limpio.
            if words_sub:
                fin_palabras = inicio + words_sub[-1]["end"] + 0.5
                if fin_palabras < fin - 2.0:  # solo si hay >2s de silencio al final
                    ahorro = fin - fin_palabras
                    fin = fin_palabras
                    nueva_dur = fin - inicio
                    clip = clip.subclipped(0, nueva_dur)
                    print(f"   ✂️ Trim fin: -{ahorro:.1f}s de silencio → duración {nueva_dur:.0f}s")

            final = nero_reframe(video_path, clip, inicio, fin)
            if words_sub and con_subs:
                final = agregar_subtitulos(final, words_sub, fuente=fuente_sub, modo=modo_sub)

            # Fade in 0.5s al inicio + fade out 0.5s al final (estilo CapCut)
            _dur = final.duration
            def _fades(get_frame, t):
                frame = get_frame(t).astype('float32')
                if t < 0.5:
                    frame *= t / 0.5
                elif t > _dur - 0.5:
                    frame *= max(0.0, (_dur - t) / 0.5)
                return frame.astype('uint8')
            final = final.transform(_fades)

            output_path = os.path.join(folder, f"KLYPO_CLIP_{clips_editados+1}_{titulo_archivo}.mp4")
            final.write_videofile(
                output_path, codec="libx264", fps=24,
                ffmpeg_params=["-preset", "ultrafast"],
                threads=4,
                logger=None
            )
            clip.close()
            final.close()
            clips_editados += 1

        except Exception as e:
            print(f"⚠️ Error editando clip {i+1}: {e}")
            continue

    video.close()
    print(f"\n🏁 ¡Misión cumplida! {clips_editados} clips en '{folder}'.")

if __name__ == "__main__":
    ejecutar_nero()
