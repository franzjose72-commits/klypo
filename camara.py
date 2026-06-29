# ─────────────────────────────────────────────────────────────────────────────
# KLYPO PIPELINE: REGLA DE RETENCIÓN DE AUDIENCIA EN SELECCIÓN DE CLIPS
# ─────────────────────────────────────────────────────────────────────────────
# Al extraer un clip ganador mediante el análisis de transcripción (timestamps):
# 1. ELIMINACIÓN DE COLA INACTIVA: El clip debe terminar de forma fulminante en la
#    última palabra de la conclusión del orador principal.
# 2. ANTI-OFF-TOPIC: Si el entrevistador hace una pregunta de transición hacia otro
#    tema, o si la conversación se desvía aunque sea por 3 segundos, corta el clip
#    inmediatamente ANTES de que empiece esa transición.
# 3. NO DEJAR A MEDIAS: Es preferible tener un clip de 45 segundos perfectamente
#    autocontenido sobre una sola idea, que un clip de 60 segundos que incluya
#    los primeros segundos de una idea que no se llega a explicar.
# ─────────────────────────────────────────────────────────────────────────────

import os, cv2, logging
import numpy as np
import moviepy as mpy
logging.getLogger("mediapipe").setLevel(logging.ERROR)
os.environ["GLOG_minloglevel"] = "3"
import mediapipe as mp
import torch
from scipy.io import wavfile
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

try:
    face_engine = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.25
    )
    print("👀 KLYPO: DETECCIÓN FACIAL ACTIVADA.")
except Exception as e:
    print(f"⚠️ MediaPipe no cargó: {e}")
    face_engine = None

# ── Silero VAD — carga lazy, corre en CPU ────────────────────────────────────
_vad_model      = None
_vad_get_speech = None

def _inicializar_vad():
    global _vad_model, _vad_get_speech
    if _vad_model is not None:
        return True
    try:
        print("   🎤 Cargando Silero VAD...")
        m, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        _vad_model      = m
        _vad_get_speech = utils[0]
        print("   ✅ Silero VAD listo (CPU)")
        return True
    except Exception as e:
        print(f"   ⚠️ Silero VAD no disponible: {e}")
        return False


def detectar_cortes_vad(video_path, inicio, fin, umbral_ms=350):
    """
    Extrae el audio del clip de video, lo analiza con Silero VAD y devuelve
    timestamps (relativos al clip, en segundos) donde comienza cada nueva
    locución tras un silencio >= umbral_ms. Son los triggers de hard cut por voz.
    """
    if not _inicializar_vad():
        return []
    temp_wav = f"_vad_{int(inicio)}_{int(fin)}.wav"
    try:
        seg = mpy.VideoFileClip(video_path).subclipped(inicio, fin)
        seg.audio.write_audiofile(
            temp_wav, fps=16000, nbytes=2, codec='pcm_s16le',
            ffmpeg_params=["-ac", "1"], logger=None
        )
        seg.close()

        rate, data = wavfile.read(temp_wav)
        if data.ndim > 1:
            data = data[:, 0]
        wav_t = torch.FloatTensor(data.astype(np.float32) / 32768.0)

        speech_ts = _vad_get_speech(
            wav_t, _vad_model, sampling_rate=16000,
            threshold=0.45,
            min_silence_duration_ms=umbral_ms,
            min_speech_duration_ms=150,
        )

        cortes = []
        for i in range(1, len(speech_ts)):
            gap_ms = (speech_ts[i]['start'] - speech_ts[i - 1]['end']) / 16.0
            if gap_ms >= umbral_ms:
                t_corte = speech_ts[i]['start'] / 16000.0
                cortes.append(t_corte)

        print(f"   🎙️ VAD: {len(speech_ts)} frases | {len(cortes)} silencios >={umbral_ms}ms")
        return cortes

    except Exception as e:
        print(f"   ⚠️ VAD [{inicio:.0f}s-{fin:.0f}s]: {e}")
        return []
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def detectar_caras_frame(frame, w, h):
    if not face_engine:
        return []
    try:
        if frame.shape[0] > 480:
            scale_f = 480 / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale_f), 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_engine.process(rgb)
        if not results.detections:
            return []
        caras = []
        for det in results.detections:
            # Score bajo (< 0.5) = falso positivo: lámparas, plantas, pósters, etc.
            if det.score[0] < 0.5:
                continue
            bbox   = det.location_data.relative_bounding_box
            cx     = int((bbox.xmin + bbox.width / 2) * w)
            cy_rel = bbox.ymin + bbox.height / 2
            bh_rel = bbox.height
            bw_rel = bbox.width
            tam    = bbox.width * bbox.height
            if cy_rel < 0.65:
                caras.append({"cx": cx, "cy_rel": cy_rel, "bh_rel": bh_rel, "bw_rel": bw_rel, "tam": tam})
        caras.sort(key=lambda x: x["tam"], reverse=True)
        return caras
    except:
        return []


def detectar_corte_camara(frame_anterior, frame_actual):
    try:
        if frame_anterior is None:
            return False
        f1   = cv2.resize(frame_anterior, (64, 36))
        f2   = cv2.resize(frame_actual,   (64, 36))
        diff = np.mean(np.abs(f1.astype(float) - f2.astype(float)))
        return diff > 30
    except:
        return False


def estabilizar_speakers(speaker_por_intervalo, min_intervalos=6):
    if not speaker_por_intervalo:
        return {}
    estabilizado = {}
    keys = sorted(speaker_por_intervalo.keys())
    i = 0
    while i < len(keys):
        speaker_actual = speaker_por_intervalo[keys[i]]
        j = i
        while j < len(keys) and speaker_por_intervalo.get(keys[j]) == speaker_actual:
            j += 1
        duracion = j - i
        if duracion >= min_intervalos:
            for k in range(i, j):
                estabilizado[keys[k]] = speaker_actual
        i = j
    return estabilizado


def precalcular_posiciones(video_path, inicio, fin):
    print(f"   🧠 Análisis de cámara V88 (cambio de orador estabilizado: 6s mínimo)...")
    duracion = fin - inicio

    cap      = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    w_video  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_video  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    intervalos = [i * 2.0 for i in range(int(duracion / 2))]

    start_frame          = int(inicio * fps_video)
    frames_por_intervalo = max(1, round(fps_video * 2.0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    caras_por_intervalo = {}
    prev_small          = None  # thumbnail 64×36: reemplaza el batch completo de frames

    print(f"   🔎 Analizando {len(intervalos)} intervalos (streaming)...")
    for idx, t_relativo in enumerate(intervalos):
        es_corte           = False
        t_corte_exacto_rel = t_relativo
        fi_corte           = 0
        frame_caras        = None
        mov_sum            = 0.0
        mov_2d_sum         = np.zeros((36, 64), dtype=np.float32)  # mapa 2D de movimiento
        mov_count          = 0
        frames_leidos      = 0
        last_frame         = None
        h, w               = h_video, w_video

        for fi in range(frames_por_intervalo):
            ret_b, f_b = cap.read()
            if not ret_b or f_b is None:
                break
            frames_leidos += 1
            if frames_leidos == 1:
                h, w, _ = f_b.shape

            curr_small = cv2.resize(f_b, (64, 36)).astype(np.float32)

            if prev_small is not None:
                diff_full = np.abs(curr_small - prev_small)
                diff_val  = float(np.mean(diff_full))

                # Detección de corte físico de cámara (mismo umbral diff > 30)
                if not es_corte and diff_val > 30:
                    es_corte           = True
                    fi_corte           = fi
                    frame_abs          = start_frame + idx * frames_por_intervalo + fi
                    t_corte_exacto_rel = max(0.0, frame_abs / fps_video - inicio)

                # Movimiento total + mapa 2D (36×64): promedio sobre canales RGB → (36, 64)
                # Preservar dimensión de filas permite discriminar boca (fila baja) de frente (fila alta)
                mov_sum    += diff_val
                mov_2d_sum += np.mean(diff_full, axis=2)
                mov_count  += 1

            prev_small = curr_small
            last_frame = f_b

        if frames_leidos == 0:
            caras_por_intervalo[idx] = {
                "caras": [], "es_corte": False,
                "w": w_video, "h": h_video,
                "t_corte_exacto_rel": t_relativo,
                "movimiento": 0.0,
                "mov_2d": np.zeros((36, 64), dtype=np.float32),
            }
            prev_small = None
            continue

        if frame_caras is None:
            frame_caras = last_frame

        caras           = detectar_caras_frame(frame_caras, w, h)
        movimiento_lote = mov_sum / max(1, mov_count)
        mov_2d_norm     = mov_2d_sum / max(1, mov_count)

        caras_por_intervalo[idx] = {
            "caras": caras, "es_corte": es_corte, "w": w, "h": h,
            "t_corte_exacto_rel": t_corte_exacto_rel,
            "movimiento": movimiento_lote,
            "mov_2d":    mov_2d_norm,
        }

    cap.release()

    # VAD desactivado: inyectar cortes de crop en planos continuos produce jump cuts sucios.
    # Los cortes de crop SOLO se disparan en cortes físicos de cámara (diff > 30).

    # Jump cut smoothing: reubicar corte en pausa si cae sobre gesto
    UMBRAL_PAUSA = 8.0
    UMBRAL_GESTO = 38.0
    VENTANA_JC   = 2

    for jc_idx in range(1, len(intervalos)):
        if not caras_por_intervalo.get(jc_idx, {}).get("es_corte"):
            continue
        mov_antes   = [caras_por_intervalo.get(jc_idx - k, {}).get("movimiento", 0.0) for k in range(1, VENTANA_JC + 1)]
        mov_despues = [caras_por_intervalo.get(jc_idx + k, {}).get("movimiento", 0.0) for k in range(0, VENTANA_JC)]
        if max(mov_antes + mov_despues, default=0.0) <= UMBRAL_GESTO:
            continue
        mejor_offset = None
        for offset in sorted(range(-VENTANA_JC, VENTANA_JC + 1), key=abs):
            if offset == 0:
                continue
            cand = jc_idx + offset
            if cand < 1 or cand >= len(intervalos):
                continue
            if caras_por_intervalo.get(cand, {}).get("es_corte"):
                continue
            if caras_por_intervalo.get(cand, {}).get("movimiento", 999.0) < UMBRAL_PAUSA:
                mejor_offset = offset
                break
        if mejor_offset is not None:
            nuevo   = jc_idx + mejor_offset
            t_nuevo = intervalos[nuevo]
            caras_por_intervalo[jc_idx]["es_corte"] = False
            caras_por_intervalo[nuevo]["es_corte"]  = True
            caras_por_intervalo[nuevo]["t_corte_exacto_rel"] = t_nuevo
            print(f"   🔄 Jump cut {jc_idx*2.0:.1f}s → {t_nuevo:.1f}s")

    # Segunda pasada: delta de posición de cara entre intervalos adyacentes.
    # La cara que más se mueve = orador activo (la cabeza se mueve al hablar).
    # Se salta en cortes de cámara: allí el delta refleja cambio de plano, no movimiento real.
    for idx in range(1, len(intervalos)):
        if caras_por_intervalo.get(idx, {}).get("es_corte"):
            continue
        prev_caras = caras_por_intervalo.get(idx - 1, {}).get("caras", [])
        curr_caras = caras_por_intervalo.get(idx, {}).get("caras", [])
        if not prev_caras or not curr_caras:
            continue
        for c in curr_caras:
            closest = min(prev_caras, key=lambda p: abs(p["cx"] - c["cx"]))
            c["delta_cx"] = abs(c["cx"] - closest["cx"])

    # Speaker tracking (diarización desactivada — preparado para futura activación)
    speaker_por_intervalo_raw = {}
    speaker_por_intervalo     = estabilizar_speakers(speaker_por_intervalo_raw, min_intervalos=3)

    orador_principal = None
    if speaker_por_intervalo:
        conteo = Counter(speaker_por_intervalo.values())
        top    = conteo.most_common(1)
        orador_principal = top[0][0] if top else None

    print(f"   🔗 Calculando posiciones de cámara...")

    speaker_cx = {}

    def registrar_speaker(speaker, cx):
        if speaker not in speaker_cx:
            speaker_cx[speaker] = []
        speaker_cx[speaker].append(cx)
        if len(speaker_cx[speaker]) > 20:
            speaker_cx[speaker].pop(0)

    def cx_de_speaker(speaker):
        if speaker in speaker_cx and speaker_cx[speaker]:
            return int(np.median(speaker_cx[speaker]))
        return None

    cx_plano_actual    = w_video // 2
    posiciones_finales = {}

    UMBRAL_MOV_CARA   = 2.0   # score mínimo boca-frente para confirmar orador (escala 0-255)
    TAM_CARA_REAL     = 0.004
    tw_check          = int(h_video * 9 / 16)
    ultimo_orador_cx  = None   # memoria: cx del último orador confirmado por movimiento

    def es_centrable(cx):
        return tw_check // 2 <= cx <= w_video - tw_check // 2

    for idx in range(len(intervalos)):
        datos     = caras_por_intervalo.get(idx, {})
        caras     = datos.get("caras", [])
        speaker_a = speaker_por_intervalo.get(idx)

        caras_val = [c for c in caras if c.get("tam", 0) >= 0.001]

        if caras_val:
            caras_reales = [c for c in caras_val if c.get("tam", 0) >= TAM_CARA_REAL]

            if len(caras_reales) >= 2 and not datos.get("es_corte"):
                mov_2d = datos.get("mov_2d", np.zeros((36, 64), dtype=np.float32))

                # Discriminar orador por ZONA DE CARA:
                # - Mitad inferior (boca/mentón) = mueve el orador al hablar
                # - Mitad superior (frente/ojos) = mueve el oyente al asentir
                # Si la cara tiene más movimiento ABAJO que ARRIBA → está hablando.
                # Si tiene movimiento uniforme arriba y abajo → está asintiendo (no habla).
                for c in caras_reales:
                    cx_col   = max(0, min(63, int(c["cx"] * 64 / w_video)))
                    bw_c     = max(2, int(c.get("bw_rel", 0.08) * 64))
                    c0       = max(0, cx_col - bw_c // 2)
                    c1       = min(64, c0 + bw_c)
                    if c0 >= c1:
                        c["mov_cara"] = 0.0
                        continue
                    cy_row   = c.get("cy_rel", 0.4) * 36
                    bh_c     = max(4, int(c.get("bh_rel", 0.12) * 36))
                    face_top = max(0,  int(cy_row - bh_c * 0.5))
                    face_bot = min(35, int(cy_row + bh_c * 0.5))
                    face_mid = (face_top + face_bot) // 2
                    cols     = mov_2d[:, c0:c1]
                    mov_low  = float(np.mean(cols[face_mid:face_bot + 1])) if face_mid <= face_bot else 0.0
                    mov_high = float(np.mean(cols[face_top:face_mid]))     if face_top  <  face_mid  else 0.0
                    # Score > 0: boca mueve más que frente → habla
                    # Score ≈ 0 o negativo: movimiento uniforme → asiente
                    c["mov_cara"] = mov_low - mov_high * 0.7

                cara_max = max(caras_reales, key=lambda c: c["mov_cara"])
                mov_max  = cara_max["mov_cara"]

                # Cara actualmente enfocada (según historial)
                cara_actual = None
                mov_actual  = -99.0
                if ultimo_orador_cx is not None:
                    cara_actual = min(caras_reales, key=lambda c: abs(c["cx"] - ultimo_orador_cx))
                    mov_actual  = cara_actual["mov_cara"]

                # Cambiar orador si:
                #   a) señal absoluta: alguien supera el umbral (escena activa)
                #   b) señal relativa: cara distinta tiene claramente MÁS boca en movimiento
                #      que el actual — funciona incluso en escenas estáticas y de voz calmada
                MARGEN_CAMBIO = 1.5
                cambiar = (
                    mov_max > UMBRAL_MOV_CARA
                    or (cara_actual is not None
                        and cara_max is not cara_actual
                        and mov_max > mov_actual + MARGEN_CAMBIO
                        and mov_max > 0.0)
                )

                if cambiar:
                    cara = cara_max
                    ultimo_orador_cx = cara["cx"]
                elif cara_actual is not None:
                    cara = cara_actual
                else:
                    caras_con_delta = [c for c in caras_reales if "delta_cx" in c]
                    if caras_con_delta:
                        cara = max(caras_con_delta, key=lambda c: c["delta_cx"])
                    else:
                        cara = max(caras_reales, key=lambda c: c["tam"])
            else:
                # 1 cara, plano con corte, o sin caras reales
                cara = max(caras_val, key=lambda c: c["tam"])
                # En plano individual: actualizar también la memoria del orador
                if len(caras_reales) == 1:
                    ultimo_orador_cx = cara["cx"]

            # Si la cara elegida fuerza un crop clampeado y hay alternativa centrable, preferirla
            if not es_centrable(cara["cx"]):
                centrables = [c for c in caras_val if es_centrable(c["cx"])]
                if centrables:
                    cara = max(centrables, key=lambda c: c["tam"])

            cx_plano_actual = cara["cx"]

        if speaker_a and caras_val:
            cara_ref = min(caras_val, key=lambda c: abs(c["cx"] - cx_plano_actual))
            registrar_speaker(speaker_a, cara_ref["cx"])

        posiciones_finales[idx] = {"cx": cx_plano_actual}

    # Build cortes_exactos — estructura mínima: {t, cx}
    # Cambios de orador: solo se confirman si el nuevo cx se mantiene estable
    # durante MIN_INTERVALOS_CAMBIO consecutivos (evita jitter/oscilación).
    # Cortes físicos de cámara siguen siendo instantáneos.
    MIN_INTERVALOS_CAMBIO = 3   # 3 × 2s = 6s hablando antes de mover la cámara
    UMBRAL_CAMBIO_CX      = 150 # px mínimos para considerar cambio de orador

    cortes_exactos   = []
    cx_ultimo_corte  = None
    cx_candidato     = None
    count_candidato  = 0

    for idx in range(len(intervalos)):
        datos = caras_por_intervalo.get(idx, {})
        pos   = posiciones_finales.get(idx, {"cx": w_video // 2})
        cx    = pos["cx"]

        es_corte_fisico = datos.get("es_corte")
        es_inicio       = idx == 0

        if es_corte_fisico or es_inicio:
            # Corte físico o inicio: instantáneo, reiniciar candidato
            t_exacto = float(datos.get("t_corte_exacto_rel", idx * 2.0)) if es_corte_fisico else float(idx * 2.0)
            cortes_exactos.append({"t": t_exacto, "cx": cx})
            cx_ultimo_corte = cx
            cx_candidato    = None
            count_candidato = 0
        elif cx_ultimo_corte is not None and abs(cx - cx_ultimo_corte) > UMBRAL_CAMBIO_CX:
            # Posible cambio de orador: acumular intervalos consecutivos en la misma zona
            if cx_candidato is not None and abs(cx - cx_candidato) < UMBRAL_CAMBIO_CX // 2:
                count_candidato += 1
            else:
                cx_candidato    = cx
                count_candidato = 1

            if count_candidato >= MIN_INTERVALOS_CAMBIO:
                # Orador confirmado: añadir corte y reiniciar
                cortes_exactos.append({"t": float((idx - MIN_INTERVALOS_CAMBIO + 1) * 2.0), "cx": cx_candidato})
                cx_ultimo_corte = cx_candidato
                cx_candidato    = None
                count_candidato = 0
        else:
            # Posición estable o vuelve al orador anterior: cancelar candidato
            cx_candidato    = None
            count_candidato = 0

    cortes_exactos.sort(key=lambda x: x["t"])
    return posiciones_finales, w_video, h_video, cortes_exactos


def nero_reframe(video_path, clip_moviepy, inicio, fin):
    posiciones, w_video, h_video, cortes_exactos = precalcular_posiciones(video_path, inicio, fin)

    cx_inicial = cortes_exactos[0]["cx"] if cortes_exactos else w_video // 2
    camara = {
        "cx_actual": cx_inicial,
        "siguiente": 1,
    }

    def reframe(get_frame, t):
        frame = get_frame(t)
        h, w, _ = frame.shape
        tw = int(h * (9 / 16))

        # Hard cut usando el timestamp real de MoviePy — sin contador interno.
        # Antes se usaba fps_contador/24 que se desincronizaba cuando agregar_subtitulos
        # llamaba get_frame(0) como prueba antes del render, desplazando todos los
        # cortes 1 frame tarde (≈42ms de lag visible).
        while camara["siguiente"] < len(cortes_exactos):
            if cortes_exactos[camara["siguiente"]]["t"] <= t:
                camara["cx_actual"] = cortes_exactos[camara["siguiente"]]["cx"]
                camara["siguiente"] += 1
            else:
                break

        cx_p    = camara["cx_actual"]
        x_start = max(0, min(cx_p - tw // 2, w - tw))
        recorte = frame[:, x_start:x_start + tw]
        if recorte.shape[1] != tw:
            recorte = cv2.resize(recorte, (tw, h))
        return recorte

    return clip_moviepy.transform(reframe)
