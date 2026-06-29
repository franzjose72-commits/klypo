"""
KLYPO VIRAL — Zoom automático por tamaño de cara
Detecta la cara en una muestra de frames del clip y devuelve un factor de zoom
fijo para acercar al sujeto si se ve pequeño o lejos.
"""

import os
import logging
import numpy as np

logging.getLogger("mediapipe").setLevel(logging.ERROR)
os.environ["GLOG_minloglevel"] = "3"

import mediapipe as mp

_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.4,
)

# Umbrales de área relativa de cara (bbox.w * bbox.h respecto al frame completo)
_ZOOM_GRANDE  = 0.12   # cara ocupa >12% → sin zoom
_ZOOM_MEDIANA = 0.04   # cara ocupa 4–12% → zoom 1.3
                        # cara ocupa <4%  → zoom 1.6 (máximo)

_FACTOR_MEDIANO = 1.3
_FACTOR_LEJOS   = 1.6


def calcular_zoom(clip, n_samples=8):
    """
    Analiza n_samples frames del clip con MediaPipe y devuelve:
      (zoom_factor, cx_rel, cy_rel)
      - zoom_factor: 1.0, 1.3 o 1.6
      - cx_rel, cy_rel: centro promedio de la cara (0.0–1.0) para centrar el crop
    Si no detecta ninguna cara devuelve (1.0, 0.5, 0.5) — sin zoom, centro del frame.
    """
    duracion = clip.duration
    tiempos  = [duracion * (i + 1) / (n_samples + 1) for i in range(n_samples)]

    areas, cxs, cys = [], [], []

    for t in tiempos:
        try:
            frame = clip.get_frame(t)                          # numpy RGB uint8
            frame = np.ascontiguousarray(frame, dtype=np.uint8)  # garantiza uint8 contiguo
            h, w  = frame.shape[:2]
            result = _detector.process(frame)
            if not result.detections:
                continue
            # La detección con mayor score
            det  = max(result.detections, key=lambda d: d.score[0])
            bbox = det.location_data.relative_bounding_box
            area = bbox.width * bbox.height
            cx   = bbox.xmin + bbox.width  / 2
            cy   = bbox.ymin + bbox.height / 2
            areas.append(area)
            cxs.append(cx)
            cys.append(cy)
        except Exception:
            continue

    if not areas:
        print(f"   ⚠️ ZOOM: no se detectó cara en {n_samples} frames → sin zoom")
        return 1.0, 0.5, 0.5

    area_prom = float(np.mean(areas))
    cx_prom   = float(np.mean(cxs))
    cy_prom   = float(np.mean(cys))

    if area_prom >= _ZOOM_GRANDE:
        factor = 1.0
    elif area_prom >= _ZOOM_MEDIANA:
        factor = _FACTOR_MEDIANO
    else:
        factor = _FACTOR_LEJOS

    print(f"   🔍 Cara detectada: área={area_prom:.3f} → zoom x{factor}")
    return factor, cx_prom, cy_prom
