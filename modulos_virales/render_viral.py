"""
KLYPO VIRAL — Renderizado V46
9:16 vertical | Zoom auto | Subtítulos configurables (karaoke / bloques / none)

Flujo de 3 pasos:
  1. MoviePy render base (ultrafast)
  2. ffmpeg: zoom + letterbox 9:16 + fades
  3. ffmpeg: subtítulos ASS quemados (karaoke o bloques, según modo elegido)
"""

import os
import time
import subprocess
import moviepy as mpy
from zoom_viral import calcular_zoom

# ─── Exportación ──────────────────────────────────────────────────────────────
FADE_IN_S     = 0.08
FADE_OUT_S    = 0.08
FPS_SALIDA    = 30
CRF_CALIDAD   = 20
PRESET_FFMPEG = "fast"
TARGET_W      = 1080
TARGET_H      = 1920   # 9:16 vertical

# ─── Subtítulos ───────────────────────────────────────────────────────────────
SUB_TAMANO       = 54             # antes 82 — tamaño tipo caption natural
SUB_PRIMARY      = "&H00FFFFFF"   # blanco: palabra hablada
SUB_SECONDARY    = "&HFF000000"   # totalmente transparente: palabras futuras invisibles
SUB_BORDE        = "&H00000000"   # negro
SUB_OUTLINE      = 2
SUB_SHADOW       = 1              # sombra suave para profundidad
SUB_MARGEN_V     = 0              # sin uso real: la posición la fija \pos por línea
GAP_SILENCIO       = 0.35
GAP_PAUSA_FUERTE   = 0.60
PALABRAS_MIN_GRUPO = 3
PALABRAS_MAX_GRUPO = 4
MAX_SPAN_GRUPO     = 0.50
DESFASE_OFFSET     = 0.45
MIN_DISPLAY_BLOQUE = 0.80   # cada bloque se ve al menos 0.8s en pantalla
MAX_DISPLAY_BLOQUE = 2.50   # cada bloque desaparece tras 2.5s máximo
OFFSET_DELAY       = 0.20   # retrasa t0 para que el subtítulo no se adelante a la voz

SUB_POSICION_PCT = 0.60            # ligeramente por debajo del centro del área de contenido

_PALABRAS_SUELTAS = {
    "y", "o", "e", "u", "ni", "pero", "sino", "aunque", "si",
    "el", "la", "los", "las", "un", "una", "de", "del", "en",
    "a", "al", "con", "por", "para", "que", "se", "le", "lo",
    "su", "sus", "mi", "tu", "ya",
    # "no" excluido: es una negación, no un conector — moverlo oculta el sentido
}

_FUENTES_ASS = {
    "Arial":      "Arial",
    "Anton":      "Anton",
    "Montserrat": "Montserrat Black",
    "BebasNeue":  "Bebas Neue",
    "Poppins":    "Poppins Black",
}
_FUENTES_TTF = {
    "Anton":      "Anton.ttf",
    "Montserrat": "MontserratBlack.ttf",
    "BebasNeue":  "BebasNeue.ttf",
    "Poppins":    "Poppins.ttf",
    # Arial es fuente del sistema — no necesita copia
}
_FUENTE_DEFAULT = "Arial"

# Ruta a fonts/ en la raíz del proyecto (un nivel arriba de modulos_virales/)
_RAIZ      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FONTS_DIR = os.path.join(_RAIZ, "fonts")


# ─── Helpers ASS ──────────────────────────────────────────────────────────────

_PUNTUACION_CORTE = {'.', ',', '?', '!', ':', ';'}

def _tiene_puntuacion(word_txt):
    """True si la palabra termina con signo de puntuación que indica pausa natural."""
    clean = word_txt.strip().rstrip('"\'»)')
    return bool(clean) and clean[-1] in _PUNTUACION_CORTE

def _ass_t(t):
    h  = int(t // 3600)
    m  = int((t % 3600) // 60)
    s  = int(t % 60)
    cs = int((t % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _agrupar_palabras(words):
    if not words:
        return []
    grupos, actual = [], []

    for i, w in enumerate(words):
        actual.append(w)
        palabra_texto = w["word"].strip()

        # REGLA 1 — puntuación corta SIEMPRE (sin importar cuántas palabras lleve)
        if _tiene_puntuacion(palabra_texto):
            grupos.append(actual)
            actual = []
            continue

        # REGLA 2 — pausa de audio corta
        if i + 1 < len(words):
            gap = words[i + 1]["start"] - w["end"]
            if gap >= GAP_SILENCIO:
                grupos.append(actual)
                actual = []
                continue

        # REGLA 3 — máximo 4 palabras
        if len(actual) >= 4:
            grupos.append(actual)
            actual = []

    if actual:
        grupos.append(actual)
    return grupos


def _limpiar_orphans(grupos):
    """
    Fusiona grupo de 1 palabra con el siguiente SOLO si no hay pausa real entre ellos.
    Si la pausa es real (>= GAP_SILENCIO): dejar la palabra sola — la pausa tiene prioridad.
    """
    if not grupos:
        return grupos
    r = []
    i = 0
    while i < len(grupos):
        g = grupos[i]
        if len(g) == 1 and i + 1 < len(grupos):
            gap = grupos[i + 1][0]["start"] - g[0]["end"]
            if gap < GAP_SILENCIO:
                r.append(g + grupos[i + 1])
                i += 2
            else:
                r.append(g)
                i += 1
        else:
            r.append(g)
            i += 1
    return r


def _dividir_grupos_largos(grupos):
    """
    Divide bloques donde la última palabra empieza > MAX_SPAN_GRUPO segundos después
    de la primera. Evita que el bloque muestre palabras con demasiado adelanto.
    """
    resultado = []
    for grupo in grupos:
        if not grupo:
            continue
        if grupo[-1]["start"] - grupo[0]["start"] <= MAX_SPAN_GRUPO:
            resultado.append(grupo)
            continue
        sub = [grupo[0]]
        for w in grupo[1:]:
            if w["start"] - sub[0]["start"] > MAX_SPAN_GRUPO:
                resultado.append(sub)
                sub = [w]
            else:
                sub.append(w)
        if sub:
            resultado.append(sub)
    return resultado


def _reubicar_colgantes(grupos):
    """
    Si la última palabra de un grupo es un conector o preposición suelta (y, o, que, de, al…),
    la traslada al inicio del siguiente grupo.
    Evita que la línea termine en palabras sin peso semántico que adelantan el siguiente pensamiento.
    """
    if len(grupos) < 2:
        return grupos
    resultado = [list(g) for g in grupos]
    for i in range(len(resultado) - 1):
        g = resultado[i]
        if len(g) <= 1:
            continue
        ultima = g[-1]["word"].strip().lower().strip('.,!?;:"\'-')
        if ultima in _PALABRAS_SUELTAS:
            resultado[i + 1].insert(0, g.pop())
    return [g for g in resultado if g]


def _grupos_desde_segmentos(words, clip_duration=None):
    """
    Forma bloques de 3-4 palabras usando segmentos Whisper como ancla de tiempo REAL.

    Los timestamps de SEGMENTO son precisos (Whisper detecta dónde empieza cada pausa).
    Los timestamps de PALABRA dentro del segmento son comprimidos e inútiles para sync.

    Límites de segmento se detectan por:
      - Marca new_seg=True en la palabra (set por el transcriptor)
      - Gap de >= 0.5s entre palabras consecutivas en tiempo Whisper (pausa grande = límite real)

    El fin de cada segmento = inicio del siguiente segmento (dato fiable).
    El fin del último segmento = clip_duration (dato fiable) o fallback al end de última palabra.
    """
    if not words:
        return []

    # Detectar límites de segmento
    seg_idx = [0]
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if words[i].get("new_seg", False) or gap >= 0.5:
            seg_idx.append(i)

    # Construir lista de segmentos
    segmentos = []
    for k, start in enumerate(seg_idx):
        end = seg_idx[k + 1] if k + 1 < len(seg_idx) else len(words)
        segmentos.append(words[start:end])

    resultado = []
    for i, seg_words in enumerate(segmentos):
        seg_t0 = seg_words[0]["start"]
        if i + 1 < len(segmentos):
            seg_t1 = segmentos[i + 1][0]["start"]   # inicio del siguiente = fin real de este
        elif clip_duration is not None:
            seg_t1 = clip_duration                   # duración real del clip para el último segmento
        else:
            seg_t1 = seg_words[-1]["end"]            # fallback: end de última palabra
        seg_dur = max(seg_t1 - seg_t0, 0.1)

        # Dividir en bloques respetando puntuación y límite de palabras
        bloques = []
        current = []
        for word in seg_words:
            current.append(word)
            clean_end = word["word"].strip().rstrip('"\'»)')
            termina_oracion = bool(clean_end) and clean_end[-1] in '.?!'
            if len(current) >= PALABRAS_MAX_GRUPO:
                bloques.append(current)
                current = []
            elif len(current) >= PALABRAS_MIN_GRUPO and termina_oracion:
                bloques.append(current)
                current = []
        if current:
            bloques.append(current)

        # Fusionar bloques de 1 sola palabra con el siguiente
        bloques_ok = []
        bi = 0
        while bi < len(bloques):
            if len(bloques[bi]) == 1 and bi + 1 < len(bloques):
                bloques_ok.append(bloques[bi] + bloques[bi + 1])
                bi += 2
            else:
                bloques_ok.append(bloques[bi])
                bi += 1

        # No terminar un bloque en artículo/preposición suelta — moverlo al inicio del siguiente
        bloques_ok = _reubicar_colgantes(bloques_ok)

        # Distribuir tiempo proporcionalmente dentro del segmento
        n = len(bloques_ok)
        for k, bloque in enumerate(bloques_ok):
            t0 = seg_t0 + (k / n) * seg_dur
            t1 = seg_t0 + ((k + 1) / n) * seg_dur
            resultado.append((bloque, t0, t1))

    return resultado


def _generar_ass(words, output_path, fuente=_FUENTE_DEFAULT, modo="karaoke", pos_y=None, mayusculas=False, clip_duration=None):
    if not words or modo == "none":
        return None

    nombre_fuente = _FUENTES_ASS.get(fuente, "Arial")
    py = int(pos_y) if pos_y is not None else TARGET_H // 2

    header = f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {TARGET_W}
PlayResY: {TARGET_H}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KLYPO,{nombre_fuente},{SUB_TAMANO},{SUB_PRIMARY},{SUB_SECONDARY},{SUB_BORDE},&H00000000,0,0,0,0,100,100,0,0,1,{SUB_OUTLINE},{SUB_SHADOW},5,60,60,{SUB_MARGEN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

    lineas = [header]

    if modo == "bloques":
        if words:
            print(f"   🕐 Primer timestamp: {words[0]['start']:.3f}s")

        grupos_ts = _grupos_desde_segmentos(words, clip_duration)
        print(f"   📝 {len(words)} palabras → {len(grupos_ts)} bloques")

        # Paso 1: t0/t1 desde timestamps de palabras individuales + OFFSET_DELAY en t0
        all_timings = []
        all_texts   = []
        for grupo, _, _ in grupos_ts:
            t0 = grupo[0]["start"] + OFFSET_DELAY
            t1 = grupo[-1]["end"]
            if t0 >= t1:
                t0 = grupo[0]["start"]  # sin offset si el bloque es muy corto
            txt = " ".join(
                w["word"].strip().upper() if mayusculas else w["word"].strip()
                for w in grupo
            )
            all_timings.append([t0, t1])
            all_texts.append(txt)

        # Paso 2: corregir solapamientos — t1[i] no puede superar t0[i+1]
        for i in range(len(all_timings) - 1):
            t0_i, t1_i  = all_timings[i]
            t0_next     = all_timings[i + 1][0]
            if t1_i > t0_next:
                t1_nuevo = t0_next - 0.02
                if t1_nuevo < t0_i:
                    t1_nuevo = t0_i + 0.05
                all_timings[i][1] = t1_nuevo

        # Paso 3: generar líneas ASS
        tiempos_ass = []
        for (t0, t1), txt in zip(all_timings, all_texts):
            pos = f"{{\\pos({TARGET_W//2},{py})}}"
            lineas.append(f"Dialogue: 0,{_ass_t(t0)},{_ass_t(t1)},KLYPO,,0,0,0,,{pos}{txt}")
            tiempos_ass.append((t0, t1))

        # Verificación final
        solapamientos = 0
        for i in range(len(tiempos_ass) - 1):
            if tiempos_ass[i][1] > tiempos_ass[i + 1][0] + 0.01:
                solapamientos += 1
                print(f"   ⚠️ SOLAPAMIENTO bloque {i}: t1={tiempos_ass[i][1]:.2f}s > t0_next={tiempos_ass[i+1][0]:.2f}s")
        if solapamientos == 0:
            print(f"   ✅ Sin solapamientos en {len(tiempos_ass)} bloques ASS")

    else:  # karaoke
        grupos = _agrupar_palabras(words)
        grupos = _dividir_grupos_largos(grupos)
        grupos = _reubicar_colgantes(grupos)
        print(f"   📝 Palabras entrada: {len(words)} | Palabras en grupos: {sum(len(g) for g in grupos)}")

        for idx, grupo in enumerate(grupos):
            pos = f"{{\\pos({TARGET_W//2},{py})}}"
            t0 = grupo[0]["start"]
            t1_natural = grupo[-1]["end"] + 0.05
            if idx + 1 < len(grupos):
                t1 = min(t1_natural, grupos[idx + 1][0]["start"])
            else:
                t1 = t1_natural
            partes = []
            for j, w in enumerate(grupo):
                if j < len(grupo) - 1:
                    dur_cs = max(1, int((grupo[j + 1]["start"] - w["start"]) * 100))
                else:
                    dur_cs = max(1, int((w["end"] - w["start"]) * 100))
                txt = w["word"].strip().upper() if mayusculas else w["word"].strip()
                partes.append(f"{{\\k{dur_cs}}}{txt}")
            lineas.append(f"Dialogue: 0,{_ass_t(t0)},{_ass_t(t1)},KLYPO,,0,0,0,,{pos}{' '.join(partes)}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    return output_path


def _quemar_subs(input_path, output_path, ass_path, fuente_sub=_FUENTE_DEFAULT):
    """
    Quema el ASS en el video.
    Copia el TTF necesario al mismo dir del ASS y corre ffmpeg con cwd=ass_dir.
    Sin fontsdir — fontconfig busca en el cwd y encuentra la fuente directamente.
    """
    import shutil

    ass_dir  = os.path.dirname(os.path.abspath(ass_path))
    ass_name = os.path.basename(ass_path)

    # Copiar TTF al mismo directorio del ASS para que fontconfig lo encuentre
    ttf_copia = None
    ttf_nombre = _FUENTES_TTF.get(fuente_sub)
    if ttf_nombre:
        ttf_origen = os.path.join(_FONTS_DIR, ttf_nombre)
        if os.path.exists(ttf_origen):
            ttf_copia = os.path.join(ass_dir, ttf_nombre)
            try:
                shutil.copy2(ttf_origen, ttf_copia)
            except Exception as e:
                print(f"   ⚠️ No se pudo copiar {ttf_nombre}: {e}")
                ttf_copia = None

    cmd = [
        "ffmpeg", "-y",
        "-i", os.path.abspath(input_path),
        "-vf", f"ass={ass_name}",
        "-c:v", "libx264", "-crf", str(CRF_CALIDAD),
        "-preset", PRESET_FFMPEG,
        "-c:a", "copy",
        os.path.abspath(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, cwd=ass_dir)

    if ttf_copia and os.path.exists(ttf_copia):
        try:
            os.remove(ttf_copia)
        except Exception:
            pass

    stderr = result.stderr.decode("utf-8", errors="ignore")
    if result.returncode != 0:
        print(f"   ⚠️ ffmpeg subs FALLÓ (returncode={result.returncode})")
        print(f"   🔧 DIAG stderr completo:\n{stderr}")
    return result.returncode == 0, stderr


# ─── Render principal ─────────────────────────────────────────────────────────

def render_viral(clip, output_path, words=None, titulo=None,
                 fuente_sub=_FUENTE_DEFAULT, modo_sub="karaoke", mayusculas=False):
    """
    Exporta el clip viral en hasta 3 pasos:
    1. Render base (MoviePy ultrafast → temp)
    2. ffmpeg: zoom automático + letterbox 9:16 + fades
    3. ffmpeg: subtítulos ASS quemados (si modo_sub != "none" y hay words)

    Si modo_sub == "none": el video se genera completo con b-roll y zoom, sin subtítulos.
    """
    carpeta = os.path.dirname(output_path)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)

    tmp = output_path.replace(".mp4", "_base.mp4")
    dur = clip.duration

    try:
        # Leer dimensiones y calcular zoom ANTES de escribir/cerrar el clip.
        # calcular_zoom() lee frames — si se llama después de clip.close() el
        # reader compartido ya está cerrado y los clips pares fallan con NoneType.
        orig_w, orig_h = clip.size
        zoom_factor, cx_rel, cy_rel = calcular_zoom(clip)

        # ── Paso 1: render base ────────────────────────────────────────────────
        clip.write_videofile(
            tmp,
            codec="libx264",
            fps=FPS_SALIDA,
            ffmpeg_params=["-preset", "ultrafast", "-crf", "26"],
            threads=4,
            logger=None,
        )
        clip.close()

        # ── Paso 2: zoom automático + letterbox 9:16 + fades ─────────────────

        if zoom_factor > 1.0:
            scaled_w = (int(TARGET_W * zoom_factor) // 2) * 2
            face_x   = int(cx_rel * scaled_w)
            crop_x   = max(0, min(face_x - TARGET_W // 2, scaled_w - TARGET_W))
            crop_x   = (crop_x // 2) * 2

            vf_parts = [
                f"scale={scaled_w}:-2",
                f"crop={TARGET_W}:ih:{crop_x}:0",
                f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:black",
                f"fade=t=in:st=0:d={FADE_IN_S}",
                f"fade=t=out:st={max(0.0, dur - FADE_OUT_S):.3f}:d={FADE_OUT_S}",
            ]
            content_top    = 0
            content_bottom = TARGET_H  # llena el frame, sin letterbox
        else:
            vf_parts = [
                f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease",
                f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:black",
                f"fade=t=in:st=0:d={FADE_IN_S}",
                f"fade=t=out:st={max(0.0, dur - FADE_OUT_S):.3f}:d={FADE_OUT_S}",
            ]
            scale_aspect   = min(TARGET_W / orig_w, TARGET_H / orig_h)
            scaled_h       = orig_h * scale_aspect
            content_top    = (TARGET_H - scaled_h) / 2
            content_bottom = content_top + scaled_h

        # posición al % del área de contenido real (ignora las barras negras)
        pos_y = content_top + (content_bottom - content_top) * SUB_POSICION_PCT

        cmd = [
            "ffmpeg", "-y", "-i", tmp,
            "-vf", ",".join(vf_parts),
            "-c:v", "libx264", "-crf", str(CRF_CALIDAD),
            "-preset", PRESET_FFMPEG,
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="ignore")
            print(f"   ❌ ffmpeg paso 2 falló:")
            print(f"      {err[-500:]}")
            for _i in range(3):
                try:
                    os.replace(tmp, output_path)
                    break
                except OSError:
                    time.sleep(0.5)
        else:
            for _i in range(3):
                try:
                    os.remove(tmp)
                    break
                except OSError:
                    time.sleep(0.5)

        # ── Paso 3: subtítulos ────────────────────────────────────────────────
        if modo_sub == "none":
            print("   💬 Sin subtítulos")
        elif not words:
            print("   ⚠️ Sin words_clip — no se generan subtítulos")
        elif not os.path.exists(output_path):
            print(f"   ⚠️ Output no existe — no se generan subtítulos")
        else:
            ass_path = os.path.join(
                os.path.dirname(os.path.abspath(output_path)),
                "klypo_subs_tmp.ass"
            )
            if _generar_ass(words, ass_path, fuente_sub, modo_sub, pos_y, mayusculas, dur):
                sub_tmp = output_path.replace(".mp4", "_sub.mp4")
                ok, err = _quemar_subs(output_path, sub_tmp, ass_path, fuente_sub)
                if ok and os.path.exists(sub_tmp):
                    os.replace(sub_tmp, output_path)
                    print(f"   💬 Subtítulos {modo_sub} | fuente: {fuente_sub} | {len(words)} palabras")
                else:
                    print(f"   ⚠️ Subtítulos fallaron (ver stderr arriba)")
                    if os.path.exists(sub_tmp):
                        try: os.remove(sub_tmp)
                        except: pass
                try:
                    os.remove(ass_path)
                except:
                    pass

    except Exception as e:
        print(f"   ⚠️ Error en render: {e}")
        if os.path.exists(tmp):
            for _i in range(3):
                try:
                    os.replace(tmp, output_path)
                    break
                except OSError:
                    time.sleep(0.5)

    return output_path
