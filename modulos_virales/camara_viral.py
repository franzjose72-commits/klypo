"""
KLYPO VIRAL — Cámara y Corte Inteligente v4
- Letterbox 9:16: el video 16:9 completo centrado en 1080×1920 con barras negras
- Zonas aburridas → aceleración 1.4x con 0.5s de contexto (no jump cut crudo)
- Salida: 1080×1920 (full screen TikTok/Reels)

Independiente de camara.py del pipeline de podcasts.
"""

import cv2
import numpy as np
import moviepy as mpy
import mediapipe as mp
import logging

logging.getLogger("mediapipe").setLevel(logging.ERROR)

try:
    _face_engine = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
except Exception:
    _face_engine = None

# ─── Constantes ───────────────────────────────────────────────────────────────
TARGET_W         = 1080    # salida 9:16 full screen
TARGET_H         = 1920
ZOOM_LETTERBOX   = 1.25   # 1.0 = video completo sin recorte; 1.25 = 25% más zoom (recorta lados)
TAM_OBJETIVO     = 0.07    # cara debe ocupar ~7% del área → plano normal
TAM_MIN_ZOOM     = 0.025   # cara < 2.5% → speaker lejos → aplicar zoom
ZOOM_MAX         = 2.5     # zoom máximo automático
INTERVALO_CARA_S = 0.5     # analizar cara cada 0.5s
CX_SUAV_VENTANA  = 5       # ventana de suavizado del cx (evita jitter)

UMBRAL_ABURRIDO  = 4.0
MIN_SEG_ABURRIDO = 3.0
SPEED_ABURRIDO   = 1.4
CONTEXTO_PRE_S   = 0.5
MIN_DUR_PARTE    = 0.5


# ─── Detección de cara (propia, no usa camara.py) ────────────────────────────

def _detectar_cara(bgr_frame, w, h):
    """Devuelve {cx, cy, tam, bw_rel} o None."""
    if not _face_engine:
        return None
    try:
        frame = bgr_frame
        if frame.shape[0] > 480:
            sc = 480 / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * sc), 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = _face_engine.process(rgb)
        if not res.detections:
            return None
        det = max(res.detections, key=lambda d: d.score[0])
        if det.score[0] < 0.5:
            return None
        bb = det.location_data.relative_bounding_box
        return {
            "cx":     int((bb.xmin + bb.width / 2) * w),
            "cy":     int((bb.ymin + bb.height / 2) * h),
            "bw_rel": bb.width,
            "tam":    bb.width * bb.height,
        }
    except Exception:
        return None


# ─── Análisis de caras a lo largo del clip ───────────────────────────────────

def calcular_posiciones_cara(video_path, inicio, fin):
    """
    Analiza caras cada INTERVALO_CARA_S segundos en el clip.
    Devuelve dict {t_relativo: {cx, tam, w, h}} para cada intervalo.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(inicio * fps))

    frames_iv = max(1, int(fps * INTERVALO_CARA_S))
    duracion  = fin - inicio
    posiciones = {}
    fi = 0

    while fi / fps <= duracion:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % frames_iv == 0:
            h, w = frame.shape[:2]
            cara = _detectar_cara(frame, w, h)
            t_rel = round(fi / fps, 3)
            posiciones[t_rel] = {
                "cx":   cara["cx"]  if cara else w // 2,
                "tam":  cara["tam"] if cara else 0.03,
                "w": w, "h": h,
                "cara": cara is not None,
            }
        fi += 1

    cap.release()
    return posiciones


# ─── Smart Crop 9:16 face-centered + auto-zoom ───────────────────────────────

def aplicar_smart_crop_9_16(clip, posiciones):
    """
    Transform final del clip viral:
    1. Calcula cx suavizado para evitar jitter entre intervalos
    2. Si la cara está lejos (tam < TAM_MIN_ZOOM) → reduce el crop window
       (más zoom) para que el orador ocupe más del encuadre vertical
    3. Recorta el frame en proporción 9:16 centrado en la cara
    4. Redimensiona a TARGET_W × TARGET_H (1080×1920) con interpolación Lanczos
    """
    if not posiciones:
        def _crop_central(get_frame, t):
            frame = get_frame(t)
            h, w = frame.shape[:2]
            tw = int(h * 9 / 16)
            x0 = (w - tw) // 2
            return cv2.resize(
                frame[:, x0:x0 + tw],
                (TARGET_W, TARGET_H),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return clip.transform(_crop_central)

    tiempos = sorted(posiciones.keys())
    t_arr   = np.array(tiempos)

    # Pre-calcular cx suavizado por posición
    cx_suaves = {}
    half = CX_SUAV_VENTANA // 2
    for i, t in enumerate(tiempos):
        vecinos = tiempos[max(0, i - half):i + half + 1]
        cx_suaves[t] = int(np.median([posiciones[tv]["cx"] for tv in vecinos]))

    def _pos_en(t):
        idx = min(int(np.searchsorted(t_arr, t)), len(tiempos) - 1)
        t_k = tiempos[idx]
        return cx_suaves[t_k], posiciones[t_k]["tam"]

    def reframe(get_frame, t):
        frame = get_frame(t)
        h, w  = frame.shape[:2]

        cx, tam = _pos_en(t)

        # Base 9:16 crop width
        tw_base = int(h * 9 / 16)

        # Auto-zoom si el orador está lejos
        if tam < TAM_MIN_ZOOM and tam > 0:
            zoom = min(ZOOM_MAX, (TAM_OBJETIVO / tam) ** 0.5)
            tw   = max(64, int(tw_base / zoom))
        else:
            tw = tw_base

        # Crop centrado en la cara, dentro de los límites del frame
        x0 = max(0, min(cx - tw // 2, w - tw))
        cropped = frame[:, x0:x0 + tw]

        # Upscale a 1080×1920 (sin bandas negras)
        return cv2.resize(
            cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4
        )

    return clip.transform(reframe)


# ─── Análisis de intensidad para jump cuts ───────────────────────────────────

SALTO_FRAMES_INTENSITY = 4  # analizar 1 de cada 4 frames para detección de zonas aburridas

def analizar_intensidad_clip(video_path, inicio, fin, ventana_s=0.5):
    cap     = cv2.VideoCapture(video_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(inicio * fps))
    fps_ef   = fps / SALTO_FRAMES_INTENSITY
    frames_v = max(1, int(fps_ef * ventana_s))
    duracion = fin - inicio

    resultados, prev_small, buf, fi = [], None, [], 0
    while fi / fps <= duracion:
        if fi % SALTO_FRAMES_INTENSITY == 0:
            ret, frame = cap.read()
            if not ret:
                break
            curr = cv2.resize(frame, (64, 36)).astype(np.float32)
            if prev_small is not None:
                buf.append(float(np.mean(np.abs(curr - prev_small))))
            if len(buf) >= frames_v:
                resultados.append((fi / fps, float(np.mean(buf))))
                buf = []
            prev_small = curr
        else:
            if not cap.grab():
                break
        fi += 1

    cap.release()
    return resultados


# ─── Segmentos activos / aburridos ───────────────────────────────────────────

def calcular_segmentos(intensidades):
    """
    Devuelve lista de {inicio, fin, aburrido} o None si todo está activo.
    Incluye CONTEXTO_PRE_S de velocidad normal antes de cada zona aburrida.
    """
    if not intensidades:
        return None

    estados = [(t, v >= UMBRAL_ABURRIDO) for t, v in intensidades]

    zonas_aburridas = []
    ini_ab = None
    for t, activo in estados:
        if not activo:
            if ini_ab is None:
                ini_ab = t
        else:
            if ini_ab is not None:
                if t - ini_ab >= MIN_SEG_ABURRIDO:
                    zonas_aburridas.append((ini_ab, t))
                ini_ab = None
    if ini_ab is not None:
        ut = estados[-1][0]
        if ut - ini_ab >= MIN_SEG_ABURRIDO:
            zonas_aburridas.append((ini_ab, ut))

    if not zonas_aburridas:
        return None

    ultimo_t = estados[-1][0] + 0.25
    segs, cursor = [], 0.0

    for ab_ini, ab_fin in zonas_aburridas:
        ctx_desde = max(cursor, ab_ini - CONTEXTO_PRE_S)
        if ctx_desde > cursor + MIN_DUR_PARTE:
            segs.append({"inicio": cursor, "fin": ctx_desde, "aburrido": False})
        if ab_ini - ctx_desde >= MIN_DUR_PARTE / 2:
            segs.append({"inicio": ctx_desde, "fin": ab_ini, "aburrido": False})
        if ab_fin - ab_ini >= MIN_DUR_PARTE:
            segs.append({"inicio": ab_ini, "fin": ab_fin, "aburrido": True})
        cursor = ab_fin

    if ultimo_t - cursor >= MIN_DUR_PARTE:
        segs.append({"inicio": cursor, "fin": ultimo_t, "aburrido": False})

    return segs if segs else None


def _acelerar(clip):
    try:
        return clip.with_speed_multiplied(SPEED_ABURRIDO)
    except AttributeError:
        try:
            return clip.multiply_speed(SPEED_ABURRIDO)
        except AttributeError:
            return clip


def aplicar_velocidad_aburrido(clip, segmentos):
    if not segmentos:
        return clip
    partes = []
    for seg in segmentos:
        t_ini = seg["inicio"]
        t_fin = min(seg["fin"], clip.duration)
        if t_fin - t_ini < MIN_DUR_PARTE:
            continue
        parte = clip.subclipped(t_ini, t_fin)
        if seg["aburrido"]:
            parte = _acelerar(parte)
        partes.append(parte)
    if not partes:
        return clip
    return partes[0] if len(partes) == 1 else mpy.concatenate_videoclips(partes)


# ─── Letterbox 9:16 ──────────────────────────────────────────────────────────

def aplicar_letterbox_9_16(clip):
    """
    Ajusta el clip 16:9 en un frame 9:16 (1080×1920) con barras negras.
    ZOOM_LETTERBOX controla el acercamiento: 1.0 = video completo, 1.25 = 25% más zoom
    (recorta un poco los lados pero el contenido se ve más grande).
    """
    _geo = {}

    def letterbox(get_frame, t):
        frame    = get_frame(t)
        h_s, w_s = frame.shape[:2]
        if (w_s, h_s) not in _geo:
            scale  = (TARGET_W / w_s) * ZOOM_LETTERBOX
            new_w  = int(w_s * scale)
            new_h  = min(int(h_s * scale), TARGET_H)
            x0     = (new_w - TARGET_W) // 2 if new_w > TARGET_W else 0
            y0     = (TARGET_H - new_h) // 2
            _geo[(w_s, h_s)] = (new_w, new_h, x0, y0)
        new_w, new_h, x0, y0 = _geo[(w_s, h_s)]

        resized  = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        cropped  = resized[:, x0:x0 + TARGET_W] if new_w > TARGET_W else resized
        h_c, w_c = cropped.shape[:2]
        canvas   = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        canvas[y0:y0 + h_c, :w_c] = cropped
        return canvas

    return clip.transform(letterbox)


# ─── Pipeline viral completo ──────────────────────────────────────────────────

def procesar_clip_viral(video_path, clip, inicio, fin, t_pico_abs, **_kwargs):
    """
    Pipeline viral v6: acelera zonas aburridas.
    El letterbox 9:16 se aplica en render_viral vía ffmpeg (100x más rápido).
    """
    print(f"   🎬 Procesando {inicio:.0f}s–{fin:.0f}s...")

    intensidades = analizar_intensidad_clip(video_path, inicio, fin)
    segmentos    = calcular_segmentos(intensidades)
    if segmentos:
        n_ab = sum(1 for s in segmentos if s["aburrido"])
        clip_base = aplicar_velocidad_aburrido(clip, segmentos)
        if n_ab:
            print(f"   ⏩ {n_ab} zona(s) lentas → {SPEED_ABURRIDO}x")
    else:
        clip_base = clip

    return clip_base  # sin letterbox — render_viral lo aplica vía ffmpeg
