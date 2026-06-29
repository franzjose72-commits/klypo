import json, os, cv2, threading
import numpy as np
import moviepy as mpy
import mediapipe as mp
import torch
from scipy.io import wavfile
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from motor import buscar_ganchos_en_segmento, transcribir_segmento, descargar_video_youtube, transcribir_clip_timestamps

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def descargar_montserrat():
    os.makedirs("fonts", exist_ok=True)
    path = "fonts/MontserratBlack.ttf"
    if os.path.exists(path):
        return path
    try:
        import requests
        url = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Black.ttf"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"✅ Montserrat Black descargada → {path}")
            return path
    except Exception as e:
        print(f"⚠️ Descarga Montserrat fallida: {e}")
    return None

_montserrat_path = descargar_montserrat()

try:
    face_engine = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.15
    )
    print("👀 KLYPO: DETECCIÓN FACIAL ACTIVADA.")
except Exception as e:
    print(f"⚠️ MediaPipe no cargó: {e}")
    face_engine = None

try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=HF_TOKEN
    )
    if torch.cuda.is_available():
        diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
        print("🎙️ KLYPO: DIARIZACIÓN ACTIVADA (GPU ⚡).")
    else:
        print("🎙️ KLYPO: DIARIZACIÓN ACTIVADA (CPU).")
except Exception as e:
    print(f"⚠️ Pyannote no cargó: {e}")
    diarization_pipeline = None

def cargar_fuente_sub(size):
    # Prioridad: The Bold Font (coloca fonts/TheBoldFont.ttf manualmente)
    # luego Montserrat Black descargada, luego Impact, luego fallbacks Windows
    intentos = []
    for bold in ["fonts/TheBoldFont.ttf", "fonts/TheBold.ttf", "fonts/theboldfont.ttf"]:
        if os.path.exists(bold):
            intentos.append(bold)
            break
    if _montserrat_path:
        intentos.append(_montserrat_path)
    intentos += [
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in intentos:
        try:
            return ImageFont.truetype(path, size)
        except:
            pass
    try:
        return ImageFont.truetype("impact.ttf", size)
    except:
        return ImageFont.load_default()

def agregar_subtitulos(clip, words_con_timestamps):
    """
    Subtítulos TikTok Viral V58:
    - The Bold Font / Montserrat Black / Impact (en ese orden)
    - 90px base, máx 2 palabras por línea, todas MAYÚSCULAS
    - Blanco puro + stroke negro sólido 6px (sin sombra, sin blur)
    - Activa: amarillo #FFFF00 + 110% tamaño — cambio instantáneo
    - Posición: 70% del alto, centrado exacto sin salirse del frame
    """
    if not words_con_timestamps:
        return clip

    try:
        sample = clip.get_frame(0)
        h_clip, w_clip = sample.shape[:2]
    except Exception:
        w_clip, h_clip = clip.size

    scale   = h_clip / 1080
    FS_BASE = max(52, round(65 * scale))
    FS_ACT  = max(60, round(75 * scale))

    COLOR_BLANCO   = (255, 255, 255)
    COLOR_AMARILLO = (255, 255, 0)     # #FFFF00
    COLOR_NEGRO    = (0, 0, 0)

    _fcache = {}
    def get_font(sz):
        if sz not in _fcache:
            _fcache[sz] = cargar_fuente_sub(max(24, sz))
        return _fcache[sz]

    _img_m  = Image.new("RGB", (max(w_clip, 1), max(h_clip, 1)))
    _draw_m = ImageDraw.Draw(_img_m)
    def medir(txt, fnt):
        try:
            bb = _draw_m.textbbox((0, 0), txt, font=fnt)
            return bb[2] - bb[0], bb[3] - bb[1]
        except:
            return len(txt) * FS_BASE // 2, FS_BASE

    # Grupos de máximo 2 palabras con sus timestamps
    grupos = []
    for i in range(0, len(words_con_timestamps), 2):
        g = words_con_timestamps[i:i + 2]
        grupos.append({"palabras": g, "t_ini": g[0]["start"], "t_fin": g[-1]["end"]})

    # Pre-reducir FS_BASE/FS_ACT hasta que NINGÚN grupo supere el 88% del ancho
    # (se evalúa el peor caso: cuando cada palabra es la activa a 110%)
    gap        = max(FS_BASE // 4, 8)
    max_w_pre  = int(w_clip * 0.88)
    fs_b, fs_a = FS_BASE, FS_ACT

    def peor_ancho(fb, fa):
        worst = 0
        for g in grupos:
            for act_i in range(len(g["palabras"])):
                total = gap * max(0, len(g["palabras"]) - 1)
                for ii, p in enumerate(g["palabras"]):
                    total += medir(p["word"].upper(), get_font(fa if ii == act_i else fb))[0]
                worst = max(worst, total)
        return worst

    while peor_ancho(fs_b, fs_a) > max_w_pre and fs_b > 48:
        fs_b = max(48, fs_b - 4)
        fs_a = round(fs_b * 1.10)

    gap = max(fs_b // 4, 8)

    def frame_sub(get_frame, t):
        frame = get_frame(t)
        fh, fw = frame.shape[:2]

        gm = next((g for g in grupos
                   if g["t_ini"] - 0.05 <= t <= g["t_fin"] + 0.35), None)
        if not gm:
            return frame

        # Palabra activa = la última cuyo start ya pasó
        activa_idx = 0
        for ii, p in enumerate(gm["palabras"]):
            if t >= p["start"] - 0.05:
                activa_idx = ii

        textos = [p["word"].upper() for p in gm["palabras"]]

        # Medir cada palabra con su fuente real
        fnts, anchos, altos = [], [], []
        for ii, txt in enumerate(textos):
            f = get_font(fs_a if ii == activa_idx else fs_b)
            aw, ah = medir(txt, f)
            fnts.append(f); anchos.append(aw); altos.append(ah)

        total_w = sum(anchos) + gap * max(0, len(anchos) - 1)
        alto    = max(altos)

        # Posición 75%, centrado exacto
        y0 = int(fh * 0.75)
        y0 = min(y0, fh - alto - 20)

        x0 = (fw - total_w) // 2
        x0 = max(10, x0)
        if x0 + total_w > fw - 10:
            x0 = max(10, fw - total_w - 10)

        # Fondo: cápsula negra semitransparente
        pad_x, pad_y = 20, 12
        bg = [max(0, x0 - pad_x), max(0, y0 - pad_y),
              min(fw, x0 + total_w + pad_x), min(fh, y0 + alto + pad_y)]

        img = Image.fromarray(frame)   # frame ya es RGB desde MoviePy
        ov  = img.copy()
        dov = ImageDraw.Draw(ov)
        try:
            dov.rounded_rectangle(bg, radius=18, fill=COLOR_NEGRO)
        except AttributeError:
            dov.rectangle(bg, fill=COLOR_NEGRO)
        img  = Image.blend(img, ov, alpha=0.55)
        draw = ImageDraw.Draw(img)

        x = x0
        for ii, txt in enumerate(textos):
            color = COLOR_AMARILLO if ii == activa_idx else COLOR_BLANCO
            ya    = y0 + max(0, (alto - altos[ii]) // 2)
            # Stroke negro sólido 6px — sin sombra, sin blur
            draw.text((x, ya), txt, font=fnts[ii],
                      fill=COLOR_NEGRO, stroke_width=6, stroke_fill=COLOR_NEGRO)
            draw.text((x, ya), txt, font=fnts[ii], fill=color)
            x += anchos[ii] + gap

        return np.array(img)   # RGB de vuelta a MoviePy

    return clip.transform(frame_sub)

def detectar_caras_frame(frame, w, h):
    if not face_engine:
        return []
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_engine.process(rgb)
        if not results.detections:
            return []
        caras = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            cx = int((bbox.xmin + bbox.width / 2) * w)
            cy_rel = bbox.ymin + bbox.height / 2
            bh_rel = bbox.height
            bw_rel = bbox.width
            tam = bbox.width * bbox.height
            caras.append({"cx": cx, "cy_rel": cy_rel, "bh_rel": bh_rel, "bw_rel": bw_rel, "tam": tam})
        caras.sort(key=lambda x: x["tam"], reverse=True)
        return caras
    except:
        return []

def detectar_corte_camara(frame_anterior, frame_actual):
    try:
        if frame_anterior is None:
            return False
        f1 = cv2.resize(frame_anterior, (64, 36))
        f2 = cv2.resize(frame_actual, (64, 36))
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
        speaker_actual = speaker_por_intervalo.get(keys[i])
        j = i
        while j < len(keys) and speaker_por_intervalo.get(keys[j]) == speaker_actual:
            j += 1
        duracion = j - i
        if duracion >= min_intervalos:
            for k in range(i, j):
                estabilizado[keys[k]] = speaker_actual
        else:
            speaker_anterior = estabilizado.get(keys[i-1]) if i > 0 else speaker_actual
            for k in range(i, j):
                estabilizado[keys[k]] = speaker_anterior
        i = j
    return estabilizado

def diarizar_audio(audio_path, inicio, fin):
    if not diarization_pipeline:
        return {}
    try:
        print(f"   🎙️ Diarizando audio...")
        sample_rate, waveform = wavfile.read(audio_path)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        else:
            waveform = waveform.T
        waveform = waveform.astype(np.float32) / 32768.0
        waveform_tensor = torch.tensor(waveform)
        audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}
        diarization = diarization_pipeline(audio_input)
        # V46: DiarizeOutput es un named tuple — probar .annotation, luego [0], luego directo
        annotation = None
        if hasattr(diarization, 'annotation'):
            annotation = diarization.annotation
        elif hasattr(diarization, '__getitem__'):
            try:
                annotation = diarization[0]
            except Exception:
                annotation = diarization
        else:
            annotation = diarization
        speaker_por_intervalo = {}
        duracion = fin - inicio
        intervalos = [i * 0.5 for i in range(int(duracion * 2))]
        for idx, t_rel in enumerate(intervalos):
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                if turn.start <= t_rel <= turn.end:
                    speaker_por_intervalo[idx] = speaker
                    break
        return speaker_por_intervalo
    except Exception as e:
        print(f"   ⚠️ Error diarizando: {e}")
        return {}

def extraer_audio_clip(video_path, inicio, fin):
    audio_path = f"temp_diar_{int(inicio)}.wav"
    try:
        clip = mpy.VideoFileClip(video_path).subclipped(inicio, fin)
        clip.audio.write_audiofile(audio_path, logger=None)
        clip.close()
        return audio_path
    except:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: Lookahead — si el frame exacto del corte no tiene speaker,
#         busca en los próximos N intervalos (2 segundos = 4 intervalos de 0.5s)
# ─────────────────────────────────────────────────────────────────────────────
def obtener_speaker_en_corte(speaker_por_intervalo, idx, rango=4):
    """
    Pyannote a veces deja vacío el intervalo justo en el corte de cámara.
    Esta función mira hacia adelante hasta `rango` intervalos para encontrar
    el speaker más cercano, evitando el retraso en re-enfocar.
    """
    for offset in range(rango):
        sp = speaker_por_intervalo.get(idx + offset)
        if sp:
            return sp
    return None

def precalcular_posiciones(video_path, inicio, fin):
    print(f"   🧠 Análisis multimodal iniciado...")
    duracion = fin - inicio

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    intervalos = [i * 0.5 for i in range(int(duracion * 2))]

    # V47: diarización en hilo paralelo — corre mientras analizamos el video
    speaker_bag = {}
    def analizar_audio():
        ap = extraer_audio_clip(video_path, inicio, fin)
        if ap:
            speaker_bag.update(diarizar_audio(ap, inicio, fin))
            if os.path.exists(ap):
                os.remove(ap)
    audio_hilo = threading.Thread(target=analizar_audio, daemon=True)
    audio_hilo.start()

    # V47: primera pasada SECUENCIAL — un solo cap.set() al inicio, cero seeks en el bucle
    # Antes: 1200 cap.set() = 10-40 min. Ahora: lectura continua = 30-60 seg.
    start_frame = int(inicio * fps_video)
    frames_por_intervalo = max(1, round(fps_video * 0.5))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    caras_por_intervalo = {}
    frame_anterior = None

    print(f"   🔎 Analizando {len(intervalos)} intervalos (secuencial)...")
    for idx, t_relativo in enumerate(intervalos):
        # Leer un intervalo de frames — sin seek, puro cap.read() secuencial
        batch = []
        for _ in range(frames_por_intervalo):
            ret_b, f_b = cap.read()
            if not ret_b or f_b is None:
                break
            batch.append(f_b)

        if not batch:
            caras_por_intervalo[idx] = {
                "caras": [], "es_corte": False,
                "w": w_video, "h": h_video,
                "t_corte_exacto_rel": t_relativo
            }
            frame_anterior = None
            continue

        h, w, _ = batch[0].shape

        # Detectar corte exacto dentro del batch sin seek adicional
        es_corte = False
        t_corte_exacto_rel = t_relativo
        fi_corte = 0

        if frame_anterior is not None:
            prev_f = frame_anterior
            for fi, f in enumerate(batch):
                if detectar_corte_camara(prev_f, f):
                    es_corte = True
                    fi_corte = fi  # primer frame del NUEVO plano → cx cambia aquí, no antes
                    frame_abs = start_frame + idx * frames_por_intervalo + fi_corte
                    t_corte_exacto_rel = max(0.0, frame_abs / fps_video - inicio)
                    break
                prev_f = f

        # Frame estable para caras: 3 frames después del corte (evita blur de transición)
        if es_corte:
            frame_caras = batch[min(fi_corte + 3, len(batch) - 1)]
        else:
            frame_caras = batch[-1]

        caras = detectar_caras_frame(frame_caras, w, h)
        caras_por_intervalo[idx] = {
            "caras": caras, "es_corte": es_corte, "w": w, "h": h,
            "t_corte_exacto_rel": t_corte_exacto_rel
        }
        frame_anterior = batch[-1]

    cap.release()

    # Esperar a que termine la diarización (ya lleva tiempo corriendo en paralelo)
    audio_hilo.join()
    speaker_por_intervalo_raw = speaker_bag
    speaker_por_intervalo = estabilizar_speakers(speaker_por_intervalo_raw, min_intervalos=6)

    orador_principal = None
    entrevistador = None
    if speaker_por_intervalo:
        conteo = Counter(speaker_por_intervalo.values())
        top = conteo.most_common(2)
        orador_principal = top[0][0]
        if len(top) >= 2:
            entrevistador = top[1][0]
        print(f"   🎤 Orador: {orador_principal} | Entrevistador: {entrevistador}")

    print(f"   🔗 Fusionando audio + video...")

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

    # PRIMERA PASADA — registrar cuando hay 1 sola cara
    for idx in range(len(intervalos)):
        datos = caras_por_intervalo.get(idx, {"caras": []})
        caras = datos.get("caras", [])
        speaker_actual = speaker_por_intervalo.get(idx)
        if len(caras) == 1 and speaker_actual:
            registrar_speaker(speaker_actual, caras[0]["cx"])

    print(f"   🗺️ Speakers: {list(speaker_cx.keys())}")

    # SEGUNDA PASADA — posición fija por plano
    posiciones_finales = {}
    ultima_cx              = w_video // 2
    cx_plano_actual        = w_video // 2
    ultimo_speaker_activo  = orador_principal
    modo_split_actual      = False
    modo_geometrico_actual = False
    cx_orador_actual = w_video // 2
    cx_ent_actual    = w_video // 2
    cy_orador_actual = 0.5
    cy_ent_actual    = 0.5
    bh_orador_actual = 0.1
    bh_ent_actual    = 0.1
    bw_orador_actual = 0.1
    bw_ent_actual    = 0.1

    for idx in range(len(intervalos)):
        datos = caras_por_intervalo.get(idx, {"caras": [], "es_corte": False})
        caras = datos.get("caras", [])
        es_corte = datos.get("es_corte", False)
        speaker_actual = speaker_por_intervalo.get(idx)

        if es_corte or idx == 0:
            speaker_en_corte = obtener_speaker_en_corte(speaker_por_intervalo, idx, rango=4)
            modo_split_actual      = False
            modo_geometrico_actual = False

            caras_efectivas = caras
            if not caras:
                for fwd in range(1, 13):   # FIX 1: lookahead extendido a 6s (era 4s)
                    next_idx = idx + fwd
                    if next_idx >= len(intervalos):
                        break
                    next_caras = caras_por_intervalo.get(next_idx, {}).get("caras", [])
                    if next_caras:
                        caras_efectivas = next_caras
                        break

            # FIX 2: umbral dirty-cut bajado a 4% — evita borrar caras válidas en planos medios
            if es_corte and idx > 0 and caras_efectivas and len(caras_efectivas) < 2:
                if caras_efectivas[0].get("tam", 1) < 0.04:
                    caras_efectivas = []

            tams_dbg = [round(c.get("tam", 0) * 100, 2) for c in caras_efectivas]
            print(f"   📐 t={idx*0.5:.1f}s: {len(caras_efectivas)} caras, tamaños: {tams_dbg}%")

            if caras_efectivas:
                # V56: Sin umbral mínimo de tamaño — cualquier cara detectada cuenta
                # El zoom agresivo en crop_zoom (80%) se encarga de ampliarlas
                caras_visibles = [c for c in caras_efectivas if c.get("tam", 0) >= 0.001]
                if len(caras_visibles) >= 2 and orador_principal and entrevistador:
                    # Sin requisito de speaker_en_corte: el corte puede pasar entre frases
                    cx_or      = cx_de_speaker(orador_principal)
                    cx_ent_known = cx_de_speaker(entrevistador)
                    # Fallback si no hay historial aún: usar posición geométrica izq/dcha
                    if not cx_or or not cx_ent_known:
                        caras_sorted = sorted(caras_visibles, key=lambda c: c["cx"])
                        cx_or      = cx_or      or caras_sorted[0]["cx"]
                        cx_ent_known = cx_ent_known or caras_sorted[-1]["cx"]
                    cara_or  = min(caras_visibles, key=lambda c: abs(c["cx"] - cx_or))
                    cara_ent = min(caras_visibles, key=lambda c: abs(c["cx"] - cx_ent_known))
                    if abs(cara_or["cx"] - cara_ent["cx"]) > w_video * 0.1:
                        modo_split_actual = True
                        cx_orador_actual   = cara_or["cx"]
                        cx_ent_actual      = cara_ent["cx"]
                        cy_orador_actual   = cara_or.get("cy_rel", 0.5)
                        cy_ent_actual      = cara_ent.get("cy_rel", 0.5)
                        bh_orador_actual   = cara_or.get("bh_rel", 0.1)
                        bh_ent_actual      = cara_ent.get("bh_rel", 0.1)
                        bw_orador_actual   = cara_or.get("bw_rel", 0.1)
                        bw_ent_actual      = cara_ent.get("bw_rel", 0.1)
                        cx_plano_actual    = cx_orador_actual
                        ultima_cx          = cx_plano_actual
                        print(f"   ✅ SPLIT activado: orador@{cx_orador_actual}px ent@{cx_ent_actual}px")

                # Fallback geométrico: caras detectadas pero muy pequeñas (<5%)
                if (not modo_split_actual and orador_principal and entrevistador
                        and caras_visibles and caras_efectivas[0].get("tam", 1) < 0.05):
                    cx_or_g  = cx_de_speaker(orador_principal) or caras_visibles[0]["cx"]
                    cx_ent_g = cx_de_speaker(entrevistador)    or caras_visibles[-1]["cx"]
                    hist_or  = len(speaker_cx.get(orador_principal, []))
                    hist_ent = len(speaker_cx.get(entrevistador, []))
                    if hist_or >= 3 and hist_ent >= 3:
                        modo_split_actual      = True
                        modo_geometrico_actual = True
                        cx_orador_actual       = cx_or_g
                        cx_ent_actual          = cx_ent_g
                        cx_plano_actual        = cx_orador_actual
                        ultima_cx              = cx_plano_actual
                        print(f"   📐 SPLIT GEOMÉTRICO (caras pequeñas, t={idx*0.5:.1f}s)")

                if not modo_split_actual:
                    if speaker_en_corte:
                        cx_conocido = cx_de_speaker(speaker_en_corte)
                        # Sin historial: buscar adelante un frame donde el speaker esté solo
                        # y usar su cx como referencia para elegir la cara correcta ahora
                        if not cx_conocido:
                            for fwd in range(1, 20):
                                fwd_idx   = idx + fwd
                                if fwd_idx >= len(intervalos): break
                                fwd_sp    = speaker_por_intervalo.get(fwd_idx)
                                fwd_caras = caras_por_intervalo.get(fwd_idx, {}).get("caras", [])
                                if fwd_sp == speaker_en_corte and len(fwd_caras) == 1:
                                    cx_conocido = fwd_caras[0]["cx"]
                                    break
                        if cx_conocido:
                            cara = min(caras_efectivas, key=lambda c: abs(c["cx"] - cx_conocido))
                            if abs(cara["cx"] - cx_conocido) > w_video * 0.30:
                                cara = caras_efectivas[0]
                            cx_candidato = cara["cx"]
                        else:
                            cx_candidato = caras_efectivas[0]["cx"]
                        registrar_speaker(speaker_en_corte, cx_candidato)
                    elif len(caras_efectivas) >= 2:
                        cx_conocido = cx_de_speaker(ultimo_speaker_activo) if ultimo_speaker_activo else None
                        if cx_conocido:
                            cara = min(caras_efectivas, key=lambda c: abs(c["cx"] - cx_conocido))
                            cx_candidato = cara["cx"]
                        else:
                            cx_candidato = caras_efectivas[0]["cx"]
                    else:
                        cx_candidato = caras_efectivas[0]["cx"]

                    if abs(cx_candidato - ultima_cx) > w_video * 0.08 or ultima_cx == w_video // 2:
                        cx_plano_actual = cx_candidato
                    else:
                        cx_plano_actual = ultima_cx
                    ultima_cx = cx_plano_actual
            else:
                # Sin caras detectadas — fallback V56: speakers en lados opuestos → split
                cx_or_fb  = cx_de_speaker(orador_principal)  if orador_principal  else None
                cx_ent_fb = cx_de_speaker(entrevistador)      if entrevistador     else None
                hist_or   = len(speaker_cx.get(orador_principal,  []))
                hist_ent  = len(speaker_cx.get(entrevistador,     []))
                if (cx_or_fb and cx_ent_fb
                        and hist_or >= 5 and hist_ent >= 5
                        and abs(cx_or_fb - cx_ent_fb) > w_video * 0.20):
                    modo_split_actual      = True
                    modo_geometrico_actual = True
                    cx_orador_actual       = cx_or_fb
                    cx_ent_actual          = cx_ent_fb
                    cx_plano_actual        = cx_orador_actual
                    ultima_cx              = cx_plano_actual
                    print(f"   🔀 SPLIT por historial (t={idx*0.5:.1f}s)")
                else:
                    cx_plano_actual = ultima_cx

        # V48: tracking dinámico dentro del plano — sigue al speaker sin esperar corte de cámara
        elif not modo_split_actual and speaker_actual and len(caras) >= 2:
            cx_speaker = cx_de_speaker(speaker_actual)
            # Sin historial: buscar adelante donde este speaker esté solo
            if not cx_speaker:
                for fwd in range(1, 10):
                    fwd_idx   = idx + fwd
                    if fwd_idx >= len(intervalos): break
                    fwd_sp    = speaker_por_intervalo.get(fwd_idx)
                    fwd_caras = caras_por_intervalo.get(fwd_idx, {}).get("caras", [])
                    if fwd_sp == speaker_actual and len(fwd_caras) == 1:
                        cx_speaker = fwd_caras[0]["cx"]
                        break
            if cx_speaker:
                cara_sp = min(caras, key=lambda c: abs(c["cx"] - cx_speaker))
                if abs(cara_sp["cx"] - cx_plano_actual) > w_video * 0.15:
                    cx_plano_actual = cara_sp["cx"]
                    ultima_cx = cx_plano_actual

        if speaker_actual and caras:
            cara_cercana = min(caras, key=lambda c: abs(c["cx"] - cx_plano_actual))
            registrar_speaker(speaker_actual, cara_cercana["cx"])
            ultimo_speaker_activo = speaker_actual

        if modo_split_actual:
            posiciones_finales[idx] = {
                "cx": cx_orador_actual,
                "modo_split": True,
                "modo_geometrico": modo_geometrico_actual,
                "cx_orador": cx_orador_actual,
                "cx_entrevistador": cx_ent_actual,
                "cy_orador": cy_orador_actual,
                "cy_entrevistador": cy_ent_actual,
                "bh_orador": bh_orador_actual,
                "bh_entrevistador": bh_ent_actual,
                "bw_orador": bw_orador_actual,
                "bw_entrevistador": bw_ent_actual
            }
        else:
            posiciones_finales[idx] = {"cx": cx_plano_actual}

    # V48: captura tanto cortes de cámara como cambios de speaker dentro del plano
    cortes_exactos = []
    cx_registrado = None
    for idx in range(len(intervalos)):
        datos = caras_por_intervalo.get(idx, {})
        pos = posiciones_finales.get(idx, {"cx": w_video // 2})
        cx = pos["cx"]

        es_corte_camara = datos.get("es_corte") or idx == 0
        es_cambio_speaker = (cx_registrado is not None and
                             abs(cx - cx_registrado) > w_video * 0.1 and
                             not pos.get("modo_split", False))

        if es_corte_camara or es_cambio_speaker:
            # Corte de cámara → timestamp exacto; cambio de speaker → timestamp del intervalo
            t_exacto = float(datos.get("t_corte_exacto_rel", idx * 0.5)) if es_corte_camara else float(idx * 0.5)
            cortes_exactos.append({
                "t": t_exacto,
                "cx": cx,
                "modo_split": pos.get("modo_split", False),
                "modo_geometrico": pos.get("modo_geometrico", False),
                "cx_orador": pos.get("cx_orador", cx),
                "cx_entrevistador": pos.get("cx_entrevistador", cx),
                "cy_orador": pos.get("cy_orador", 0.5),
                "cy_entrevistador": pos.get("cy_entrevistador", 0.5),
                "bh_orador": pos.get("bh_orador", 0.1),
                "bh_entrevistador": pos.get("bh_entrevistador", 0.1),
                "bw_orador": pos.get("bw_orador", 0.1),
                "bw_entrevistador": pos.get("bw_entrevistador", 0.1)
            })
            cx_registrado = cx

        if cx_registrado is None:
            cx_registrado = cx

    cortes_exactos.sort(key=lambda x: x["t"])

    return posiciones_finales, w_video, h_video, cortes_exactos

def nero_reframe(video_path, clip_moviepy, inicio, fin):
    posiciones, w_video, h_video, cortes_exactos = precalcular_posiciones(video_path, inicio, fin)

    cx_inicial = cortes_exactos[0]["cx"] if cortes_exactos else w_video // 2
    primer_corte = cortes_exactos[0] if cortes_exactos else {}
    camara = {
        "cx_actual":        cx_inicial,
        "fps_contador":     0,
        "siguiente":        1,
        "modo_split":       primer_corte.get("modo_split",       False),
        "modo_geometrico":  primer_corte.get("modo_geometrico",  False),
        "cx_orador":        primer_corte.get("cx_orador",        cx_inicial),
        "cx_entrevistador": primer_corte.get("cx_entrevistador", cx_inicial),
        "cy_orador":        primer_corte.get("cy_orador",        0.5),
        "cy_entrevistador": primer_corte.get("cy_entrevistador", 0.5),
        "bh_orador":        primer_corte.get("bh_orador",        0.1),
        "bh_entrevistador": primer_corte.get("bh_entrevistador", 0.1),
        "bw_orador":        primer_corte.get("bw_orador",        0.1),
        "bw_entrevistador": primer_corte.get("bw_entrevistador", 0.1),
    }
    fps_clip = 24

    def reframe(frame):
        h, w, _ = frame.shape
        tw = int(h * (9 / 16))

        t_actual = camara["fps_contador"] / fps_clip
        camara["fps_contador"] += 1

        # Corte 1 frame antes — elimina fotograma sucio por desfase de timestamps
        margen = 1.0 / fps_clip
        while camara["siguiente"] < len(cortes_exactos):
            if cortes_exactos[camara["siguiente"]]["t"] <= t_actual + margen:
                corte = cortes_exactos[camara["siguiente"]]
                camara["cx_actual"]        = corte["cx"]
                camara["modo_split"]       = corte.get("modo_split",       False)
                camara["modo_geometrico"]  = corte.get("modo_geometrico",  False)
                camara["cx_orador"]        = corte.get("cx_orador",        corte["cx"])
                camara["cx_entrevistador"] = corte.get("cx_entrevistador", corte["cx"])
                camara["cy_orador"]        = corte.get("cy_orador",        0.5)
                camara["cy_entrevistador"] = corte.get("cy_entrevistador", 0.5)
                camara["bh_orador"]        = corte.get("bh_orador",        0.1)
                camara["bh_entrevistador"] = corte.get("bh_entrevistador", 0.1)
                camara["bw_orador"]        = corte.get("bw_orador",        0.1)
                camara["bw_entrevistador"] = corte.get("bw_entrevistador", 0.1)
                camara["siguiente"] += 1
            else:
                break

        if camara["modo_split"]:
            h2  = h // 2
            sep = np.zeros((2, tw, 3), dtype=np.uint8)

            if camara["modo_geometrico"]:
                # Split geométrico: izq/dcha según historial de posición del orador
                w2 = w // 2
                if camara["cx_orador"] <= w // 2:
                    src_top, src_bot = frame[:, :w2], frame[:, w2:]
                else:
                    src_top, src_bot = frame[:, w2:], frame[:, :w2]
                top = cv2.resize(src_top, (tw, h2))
                bot = cv2.resize(src_bot, (tw, h2))
            else:
                # Split por caras: bbox expandida 80%, aspect ratio tw:h2
                aspect = tw / max(h2, 1)

                def crop_zoom(cx_p, cy_r, bh_r, bw_r):
                    cy_abs = int(cy_r * h)
                    face_w = bw_r * w
                    face_h = bh_r * h
                    exp    = 0.8
                    cw_raw = face_w * (1 + 2 * exp)
                    ch_raw = face_h * (1 + 2 * exp)
                    if cw_raw / max(ch_raw, 1) > aspect:
                        ch_raw = cw_raw / aspect
                    else:
                        cw_raw = ch_raw * aspect
                    cw = max(1, min(int(cw_raw), w))
                    ch = max(1, min(int(ch_raw), h))
                    x1 = max(0, min(cx_p - cw // 2, w - cw))
                    y1 = max(0, min(cy_abs - ch // 2, h - ch))
                    return cv2.resize(frame[y1:y1 + ch, x1:x1 + cw], (tw, h2))

                top = crop_zoom(camara["cx_orador"],        camara["cy_orador"],
                                camara["bh_orador"],        camara["bw_orador"])
                bot = crop_zoom(camara["cx_entrevistador"], camara["cy_entrevistador"],
                                camara["bh_entrevistador"], camara["bw_entrevistador"])

            return np.vstack([top, sep, bot])

        # Single person: crop horizontal centrado en la cara
        cx_p    = camara["cx_actual"]
        x_start = cx_p - tw // 2
        x_start = max(0, min(x_start, w - tw))
        x_end   = x_start + tw
        recorte = frame[:, x_start:x_end]
        if recorte.shape[1] != tw:
            recorte = cv2.resize(recorte, (tw, h))
        return recorte

    return clip_moviepy.image_transform(reframe)

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
    print("🚀 KLYPO.IA V70: Speaker tracking con lookahead futuro cuando no hay historial.")
    link = input("🔗 Link del podcast: ")
    video_path = descargar_video_youtube(link)
    if not video_path:
        print("❌ No se pudo descargar. Abortando.")
        return

    video = mpy.VideoFileClip(video_path)
    duracion_total = video.duration
    guion_ganador = []

    print("🧠 KLYPO analizando desde el minuto 5...")
    for start in range(300, int(duracion_total), 600):
        fin_segmento = min(start + 600, duracion_total)
        temp_audio = f"temp_{start}.mp3"
        try:
            video.audio.subclipped(start, fin_segmento).write_audiofile(
                temp_audio, bitrate="16k", logger=None
            )
            texto = transcribir_segmento(temp_audio)
            if texto:
                res = buscar_ganchos_en_segmento(texto, start, duracion_total - 5)
                clips = extraer_json_de_texto(res)
                for clip in clips:
                    try:
                        clip['inicio'] = float(clip['inicio']) + start
                        clip['fin'] = float(clip['fin']) + start
                        if clip['inicio'] < 300:
                            clip['inicio'] = 300
                        guion_ganador.append(clip)
                    except:
                        continue
        except Exception as e:
            print(f"⚠️ Error en segmento {start}s: {e}")
        finally:
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

    folder = "CLIPS_KLYPO_V70"
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

            inicio = max(300, inicio)
            fin = min(fin, duracion_total - 1)

            titulo = info.get('titulo', 'Clip_Viral')
            titulo_archivo = "".join(c for c in titulo if c.isalnum() or c in (' ', '_', '-')).strip()[:50]

            print(f"✨ Editando Clip {clips_editados+1}: '{titulo}' ({duracion:.0f}s)")

            clip = video.subclipped(inicio, fin)

            # Subtítulos: extraer audio del clip y transcribir con timestamps de Whisper
            words_sub = []
            audio_sub_path = f"temp_sub_{clips_editados}.mp3"
            try:
                if clip.audio:
                    clip.audio.write_audiofile(audio_sub_path, bitrate="128k", logger=None)
                    words_sub = transcribir_clip_timestamps(audio_sub_path)
                    print(f"   📝 {len(words_sub)} palabras para subtítulos" if words_sub else "   ⚠️ Whisper devolvió 0 palabras")
            except Exception as e:
                import traceback
                print(f"   ❌ Error en subtítulos: {e}")
                traceback.print_exc()
            finally:
                if os.path.exists(audio_sub_path):
                    os.remove(audio_sub_path)

            final = nero_reframe(video_path, clip, inicio, fin)
            if words_sub:
                final = agregar_subtitulos(final, words_sub)

            output_path = os.path.join(folder, f"KLYPO_CLIP_{clips_editados+1}_{titulo_archivo}.mp4")
            final.write_videofile(output_path, codec="libx264", fps=24)
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