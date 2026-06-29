import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

_FUENTES_MAP = {
    "montserrat": ["fonts/MontserratBlack.ttf", "fonts/Montserrat.ttf"],
    "anton":      ["fonts/Anton.ttf"],
    "bebas":      ["fonts/BebasNeue.ttf"],
    "impact":     ["C:/Windows/Fonts/impact.ttf"],
}
_FALLBACKS = [
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/ariblk.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

def cargar_fuente_sub(size, fuente="montserrat"):
    intentos = _FUENTES_MAP.get(fuente, []) + _FALLBACKS
    for path in intentos:
        try:
            return ImageFont.truetype(path, size)
        except:
            pass
    return ImageFont.load_default()


def agregar_subtitulos(clip, words_con_timestamps, fuente="montserrat", modo="karaoke"):
    """
    Subtítulos TikTok — dos modos:
    - karaoke: 2 palabras por grupo, palabra activa en AMARILLO + 110% tamaño
    - bloques:  3-4 palabras por grupo, todas BLANCAS a la vez (sin resaltado)
    Fuentes: montserrat | anton | bebas | impact
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
    COLOR_AMARILLO = (255, 255, 0)
    COLOR_NEGRO    = (0, 0, 0)

    _fcache = {}
    def get_font(sz):
        if sz not in _fcache:
            _fcache[sz] = cargar_fuente_sub(max(24, sz), fuente)
        return _fcache[sz]

    _img_m  = Image.new("RGB", (max(w_clip, 1), max(h_clip, 1)))
    _draw_m = ImageDraw.Draw(_img_m)
    def medir(txt, fnt):
        try:
            bb = _draw_m.textbbox((0, 0), txt, font=fnt)
            return bb[2] - bb[0], bb[3] - bb[1]
        except:
            return len(txt) * FS_BASE // 2, FS_BASE

    # ── Construir grupos según modo ──────────────────────────────────────────
    palabras_por_grupo = 4 if modo == "bloques" else 2
    grupos = []
    for i in range(0, len(words_con_timestamps), palabras_por_grupo):
        g = words_con_timestamps[i:i + palabras_por_grupo]
        grupos.append({"palabras": g, "t_ini": g[0]["start"], "t_fin": g[-1]["end"]})

    # Pre-reducir tamaño hasta que ningún grupo supere el 88% del ancho
    gap       = max(FS_BASE // 4, 8)
    max_w_pre = int(w_clip * 0.88)
    fs_b, fs_a = FS_BASE, FS_ACT

    def peor_ancho(fb, fa):
        worst = 0
        for g in grupos:
            if modo == "bloques":
                total = gap * max(0, len(g["palabras"]) - 1) + sum(
                    medir(p["word"].upper(), get_font(fb))[0] for p in g["palabras"]
                )
                worst = max(worst, total)
            else:
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

    # ── Pre-renderizar overlays ──────────────────────────────────────────────
    def render_overlay(g, activa_idx):
        # KARAOKE: solo la palabra activa — nada más visible
        if modo == "karaoke":
            txt = g["palabras"][activa_idx]["word"].upper()
            f   = get_font(fs_a)
            aw, ah = medir(txt, f)
            y0 = min(int(h_clip * 0.75), h_clip - ah - 20)
            x0 = max(10, (w_clip - aw) // 2)
            pad_x, pad_y = 24, 14
            bg = [max(0, x0 - pad_x), max(0, y0 - pad_y),
                  min(w_clip, x0 + aw + pad_x), min(h_clip, y0 + ah + pad_y)]
            overlay = Image.new("RGBA", (w_clip, h_clip), (0, 0, 0, 0))
            draw    = ImageDraw.Draw(overlay)
            try:
                draw.rounded_rectangle(bg, radius=20, fill=(*COLOR_NEGRO, 170))
            except AttributeError:
                draw.rectangle(bg, fill=(*COLOR_NEGRO, 170))
            draw.text((x0, y0), txt, font=f,
                      fill=(*COLOR_NEGRO, 255), stroke_width=6, stroke_fill=(*COLOR_NEGRO, 255))
            draw.text((x0, y0), txt, font=f, fill=(*COLOR_AMARILLO, 255))
            return np.array(overlay)

        # BLOQUES: todas las palabras juntas, todas blancas
        textos = [p["word"].upper() for p in g["palabras"]]
        fnts, anchos, altos = [], [], []
        for txt in textos:
            f = get_font(fs_b)
            aw, ah = medir(txt, f)
            fnts.append(f); anchos.append(aw); altos.append(ah)

        total_w = sum(anchos) + gap * max(0, len(anchos) - 1)
        alto    = max(altos)
        y0 = min(int(h_clip * 0.75), h_clip - alto - 20)
        x0 = max(10, (w_clip - total_w) // 2)
        if x0 + total_w > w_clip - 10:
            x0 = max(10, w_clip - total_w - 10)

        pad_x, pad_y = 20, 12
        bg = [max(0, x0 - pad_x), max(0, y0 - pad_y),
              min(w_clip, x0 + total_w + pad_x), min(h_clip, y0 + alto + pad_y)]

        overlay = Image.new("RGBA", (w_clip, h_clip), (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)
        try:
            draw.rounded_rectangle(bg, radius=18, fill=(*COLOR_NEGRO, 140))
        except AttributeError:
            draw.rectangle(bg, fill=(*COLOR_NEGRO, 140))

        x = x0
        for ii, txt in enumerate(textos):
            ya = y0 + max(0, (alto - altos[ii]) // 2)
            draw.text((x, ya), txt, font=fnts[ii],
                      fill=(*COLOR_NEGRO, 255), stroke_width=6, stroke_fill=(*COLOR_NEGRO, 255))
            draw.text((x, ya), txt, font=fnts[ii], fill=(*COLOR_BLANCO, 255))
            x += anchos[ii] + gap

        return np.array(overlay)

    # Bloques: un overlay por grupo. Karaoke: un overlay por (grupo, palabra_activa)
    pre_overlays = {}
    for gi, g in enumerate(grupos):
        if modo == "bloques":
            pre_overlays[(gi, 0)] = render_overlay(g, 0)
        else:
            for ai in range(len(g["palabras"])):
                pre_overlays[(gi, ai)] = render_overlay(g, ai)

    group_times = [(gi, g["t_ini"], g["t_fin"]) for gi, g in enumerate(grupos)]

    def frame_sub(get_frame, t):
        frame = get_frame(t)

        gi_active = None
        for gi, t_ini, t_fin in group_times:
            if t_ini <= t <= t_fin + 0.10:
                gi_active = gi
        if gi_active is None:
            return frame

        if modo == "bloques":
            overlay = pre_overlays[(gi_active, 0)]
        else:
            g  = grupos[gi_active]
            ai = 0
            for ii, p in enumerate(g["palabras"]):
                if t >= p["start"]:
                    ai = ii
            overlay = pre_overlays[(gi_active, ai)]

        alpha  = overlay[:, :, 3:4].astype(np.float32) / 255.0
        rgb    = overlay[:, :, :3].astype(np.float32)
        result = np.clip(
            frame.astype(np.float32) * (1.0 - alpha) + rgb * alpha,
            0, 255
        ).astype(np.uint8)
        return result

    return clip.transform(frame_sub)
