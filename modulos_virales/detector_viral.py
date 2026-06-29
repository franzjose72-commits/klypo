"""
KLYPO VIRAL — Detector de Momentos de Impacto v3
Detecta picos de audio, movimiento y palabras clave.
Sin generación de títulos ni lógica de zoom (eso vive en camara_viral.py).

Independiente del pipeline de podcasts.
"""

import os
import re
import json
import time
import subprocess
import numpy as np
import cv2
from scipy.io import wavfile
from scipy.signal import find_peaks
import moviepy as mpy
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

try:
    _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception:
    _groq = None

PALABRAS_IMPACTO = {
    "nunca", "jamas", "jamás", "siempre", "imposible", "increible", "increíble",
    "brutal", "locura", "mentira", "secreto", "viral", "record", "récord",
    "historico", "histórico", "primera vez", "nadie sabe", "nunca visto",
    "wow", "boom", "crack", "epico", "épico", "insane", "crazy", "alucinante",
    "salvaje", "tremendo", "impresionante", "alucina", "no puede ser",
    "te juro", "literalmente", "espera", "ojo", "atencion", "atención",
    "escucha", "mira",
}


def _extraer_wav_temp(video_path, output="__viral_audio.wav"):
    clip = mpy.VideoFileClip(video_path)
    clip.audio.write_audiofile(
        output, fps=16000, nbytes=2,
        ffmpeg_params=["-ac", "1"], logger=None
    )
    clip.close()
    return output


# ─── 1. PICOS DE VOLUMEN ─────────────────────────────────────────────────────

def detectar_picos_audio(video_path, umbral_factor=2.2, ventana_s=0.1):
    """Devuelve lista de (timestamp, intensidad) con picos de volumen."""
    temp = "__viral_picos_audio.wav"
    try:
        _extraer_wav_temp(video_path, temp)
        rate, data = wavfile.read(temp)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32) / 32768.0

        n = int(rate * ventana_s)
        n_win = len(data) // n
        rms = np.array([
            np.sqrt(np.mean(data[i * n:(i + 1) * n] ** 2))
            for i in range(n_win)
        ])

        umbral = np.mean(rms) * umbral_factor
        peaks, _ = find_peaks(rms, height=umbral, distance=int(0.8 / ventana_s))
        return [(float(p * ventana_s), float(rms[p])) for p in peaks]

    except Exception as e:
        print(f"   ⚠️ Picos audio: {e}")
        return []
    finally:
        if os.path.exists(temp):
            os.remove(temp)


# ─── 2. PICOS DE MOVIMIENTO ──────────────────────────────────────────────────

def detectar_picos_movimiento(video_path, ventana_s=0.5, umbral=20.0):
    """
    Detecta movimiento via pipe ffmpeg a 1fps — rápido para videos largos.
    18 min → solo ~1100 frames en lugar de ~33000.
    """
    FPS_M = 1        # 1 frame por segundo es suficiente para detectar picos
    W, H  = 64, 36
    SIZE  = W * H * 3

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"scale={W}:{H},fps={FPS_M}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames_v = max(1, int(FPS_M * ventana_s))
    picos, prev, buf, fi = [], None, [], 0

    while True:
        data = proc.stdout.read(SIZE)
        if len(data) < SIZE:
            break
        curr = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3).astype(np.float32)
        if prev is not None:
            buf.append(float(np.mean(np.abs(curr - prev))))
        if len(buf) >= frames_v:
            v = float(np.mean(buf))
            t = fi / FPS_M
            if v > umbral:
                picos.append((t, v))
            buf = []
        prev = curr
        fi += 1

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    return picos


# ─── 3. PALABRAS CLAVE ───────────────────────────────────────────────────────

def detectar_palabras_clave(video_path):
    """
    Transcribe con Whisper y busca palabras de impacto.
    Devuelve {keywords: [(t, texto, score)], words: [{word, start, end}]}
    """
    if not _groq:
        print("   ⚠️ Groq no disponible.")
        return {"keywords": [], "words": []}

    temp = "__viral_kw.mp3"
    try:
        clip = mpy.VideoFileClip(video_path)
        clip.audio.write_audiofile(temp, bitrate="32k", logger=None)
        clip.close()

        resp = None
        for _ in range(3):
            try:
                with open(temp, "rb") as f:
                    resp = _groq.audio.transcriptions.create(
                        file=(temp, f.read()),
                        model="whisper-large-v3",
                        response_format="verbose_json",
                        timestamp_granularities=["word"]
                    )
                break
            except Exception as e:
                if "429" in str(e):
                    print("   ⏳ Rate limit Groq, esperando 60s...")
                    time.sleep(60)
                else:
                    print(f"   ⚠️ Whisper: {e}")
                    return {"keywords": [], "words": []}

        if not resp:
            return {"keywords": [], "words": []}

        raw = (resp.get("words") if isinstance(resp, dict)
               else getattr(resp, "words", None)) or []

        words, keywords = [], []
        for w in raw:
            texto = (w.get("word", "") if isinstance(w, dict) else getattr(w, "word", "")).strip()
            t_s   = float(w.get("start", 0) if isinstance(w, dict) else getattr(w, "start", 0))
            t_e   = float(w.get("end",   0) if isinstance(w, dict) else getattr(w, "end",   0))
            if not texto:
                continue
            words.append({"word": texto, "start": t_s, "end": t_e})
            tl = texto.lower()
            for kw in PALABRAS_IMPACTO:
                if kw in tl:
                    keywords.append((t_s, texto, 1.5))
                    break

        print(f"   💬 {len(keywords)} palabras de impacto encontradas")
        return {"keywords": keywords, "words": words}

    except Exception as e:
        print(f"   ⚠️ Keywords: {e}")
        return {"keywords": [], "words": []}
    finally:
        if os.path.exists(temp):
            os.remove(temp)


# ─── 4. FUSIÓN EN SEGMENTOS ──────────────────────────────────────────────────

MIN_GAP_CLIPS     = 45   # segundos mínimos entre el pico de un clip y el siguiente
MAX_CLIPS_VIRALES = 8    # máximo de clips por video — evita cortar TODO el video en trozos

def fusionar_en_clips(picos_audio, picos_video, palabras_clave,
                      duracion_total, dur_min=15, dur_max=60, ventana_fusion=2.0):
    """
    Une picos en segmentos de clip. Solo mantiene los más intensos y separados:
    - Mínimo MIN_GAP_CLIPS segundos entre picos (evita clips consecutivos)
    - Máximo MAX_CLIPS_VIRALES clips en total
    """
    todos = []
    for t, v in picos_audio:
        todos.append({"t": t, "score": v * 2.5,  "tipo": "audio"})
    for t, v in picos_video:
        todos.append({"t": t, "score": v / 8.0,  "tipo": "video"})
    for t, _, s in palabras_clave:
        todos.append({"t": t, "score": s,         "tipo": "palabra"})

    if not todos:
        return []

    todos.sort(key=lambda x: x["t"])
    grupos, grupo = [], [todos[0]]
    for m in todos[1:]:
        if m["t"] - grupo[-1]["t"] <= ventana_fusion:
            grupo.append(m)
        else:
            grupos.append(grupo)
            grupo = [m]
    grupos.append(grupo)

    segmentos = []
    for g in grupos:
        t_pico = float(np.mean([m["t"] for m in g]))
        score  = sum(m["score"] for m in g)
        tipos  = "+".join(sorted({m["tipo"] for m in g}))
        margen = min(8.0, dur_max * 0.25)
        inicio = max(0.0, t_pico - margen)
        fin    = min(duracion_total, inicio + dur_max)
        inicio = max(0.0, fin - dur_max)
        if fin - inicio < dur_min:
            continue
        segmentos.append({"inicio": inicio, "fin": fin,
                          "intensidad": score, "tipo": tipos, "t_pico": t_pico})

    # Ordenar por intensidad → los más virales primero
    segmentos.sort(key=lambda x: x["intensidad"], reverse=True)

    limpios = []
    for seg in segmentos:
        # Descartar si solapa >40% con un clip ya aceptado
        solapado = any(
            min(seg["fin"], ex["fin"]) - max(seg["inicio"], ex["inicio"])
            > (seg["fin"] - seg["inicio"]) * 0.4
            for ex in limpios
        )
        # Descartar si el pico está demasiado cerca de otro ya aceptado
        muy_cerca = any(
            abs(seg["t_pico"] - ex["t_pico"]) < MIN_GAP_CLIPS
            for ex in limpios
        )
        if not solapado and not muy_cerca:
            limpios.append(seg)
        if len(limpios) >= MAX_CLIPS_VIRALES:
            break

    limpios.sort(key=lambda x: x["inicio"])
    print(f"   📍 {len(limpios)} momentos virales detectados (gap ≥{MIN_GAP_CLIPS}s)")
    return limpios


# ─── 5. TRANSCRIPCIÓN COMPLETA (para modo IA) ────────────────────────────────

def transcribir_video_completo(video_path, duracion_max_s=1500):
    """
    Transcribe el video con AssemblyAI (word-level timestamps en segundos).
    Retorna {"text": str para Llama, "words": [{"word","start","end","new_seg"}] para subtítulos}.
    Formato de salida IDÉNTICO al anterior — motor_viral.py no necesita cambios.
    """
    import assemblyai as aai

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("   ⚠️ ASSEMBLYAI_API_KEY no encontrada en .env")
        return {"text": "", "words": []}

    aai.settings.api_key = api_key
    temp = "__viral_full_tx.mp3"
    try:
        # Exportar audio del video
        clip = mpy.VideoFileClip(video_path)
        dur  = min(clip.duration, duracion_max_s)
        clip.subclipped(0, dur).audio.write_audiofile(temp, bitrate="64k", logger=None)
        clip.close()

        # Transcribir con AssemblyAI
        config = aai.TranscriptionConfig(
            language_code="es",
            punctuate=True,
            format_text=True,
        )
        transcript = aai.Transcriber().transcribe(temp, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            print(f"   ⚠️ AssemblyAI error: {transcript.error}")
            return {"text": "", "words": []}

        # Texto para Llama — formato "t=X.Xs: texto" por oración (equivalente a segmentos Whisper)
        sentences = transcript.get_sentences()
        lineas = [
            f"t={sent.start / 1000:.1f}s: {sent.text.strip()}"
            for sent in sentences
            if sent.text.strip()
        ]

        # El inicio de cada oración marca new_seg (equivale a las fronteras de segmento de Whisper)
        # AssemblyAI: sent.start == primer word.start de esa oración → comparación exacta en ms
        sentence_starts_ms = {sent.start for sent in sentences}

        # Words — AssemblyAI da timestamps en milisegundos → convertir a segundos
        words = []
        for w in (transcript.words or []):
            texto = (w.text or "").strip()
            if not texto:
                continue
            ws = w.start / 1000.0   # ms → s
            we = w.end   / 1000.0   # ms → s
            is_new_seg = w.start in sentence_starts_ms
            words.append({"word": texto, "start": ws, "end": we, "new_seg": is_new_seg})

        print(f"   ✅ {len(words)} palabras | {len(sentences)} oraciones detectadas")

        words = _limpiar_disfluencias(words)
        return {"text": "\n".join(lineas), "words": words}

    except Exception as e:
        print(f"   ⚠️ Transcripción AssemblyAI: {e}")
        return {"text": "", "words": []}
    finally:
        if os.path.exists(temp):
            try: os.remove(temp)
            except: pass


def _limpiar_disfluencias(words):
    """
    Elimina la PRIMERA ocurrencia de repeticiones consecutivas idénticas ('que que' → 'que').
    Mantiene la SEGUNDA ocurrencia (timestamp más tardío = más preciso).
    NUNCA elimina palabras distintas. NUNCA cambia el orden cronológico.
    """
    if not words:
        return words
    clean = []
    i = 0
    while i < len(words):
        curr  = words[i]["word"].strip().lower().strip('.,!?;:"\'')
        nxt   = words[i + 1]["word"].strip().lower().strip('.,!?;:"\'') if i + 1 < len(words) else None
        if curr == nxt:
            # Skip primera ocurrencia, procesar segunda en la siguiente iteración
            i += 1
            continue
        clean.append(words[i])
        i += 1
    n_removed = len(words) - len(clean)
    if n_removed:
        print(f"   🧹 {n_removed} disfluencias eliminadas ('que que', etc.)")
    # Verificar orden cronológico (soft check — Whisper a veces tiene timestamps no monotónicos)
    if len(clean) > 1:
        malos = [j for j in range(len(clean) - 1) if clean[j]["start"] > clean[j + 1]["start"]]
        if malos:
            print(f"   ⚠️ {len(malos)} timestamps no monotónicos en Whisper — subtítulos pueden desfasarse levemente")
    return clean


# ─── 6. DETECCIÓN IA — Llama detecta TODOS los clips ganadores ───────────────

_PROMPT_CLIPS_IA = """Eres el mejor editor de clips virales del mundo. Tu trabajo es detectar TODOS los momentos ganadores de esta transcripción.

UN CLIP GANADOR es un momento donde el orador dice algo que hace que la gente SE QUEDE A ESCUCHAR porque:
→ Es una verdad que nadie suele decir en voz alta
→ Es una reflexión profunda o personal que resuena ("eso me pasa a mí")
→ Es un dato o revelación que sorprende o contradice lo que creías
→ Es una historia real con tensión, fracaso, o aprendizaje duro
→ Es algo polémico, controvertido, o que genera debate
→ Es humor que viene de una verdad incómoda

━━━ LEY DEL INICIO — EL CLIP DEBE TENER CONTEXTO ━━━
El espectador llega al clip SIN haber visto nada antes. Debe entender de qué va en los primeros 3 segundos.

PASO 1 — Identifica el momento más potente del tema (el remate, la revelación, la frase fuerte).
PASO 2 — Retrocede hasta el INICIO de la oración o pregunta que da contexto a ese momento.
PASO 3 — Ese es tu timestamp "inicio". Nunca empieces a mitad de frase.

INICIO CORRECTO:
✓ Empieza en el inicio de una oración completa: "Yo perdí todo en 6 meses porque..."
✓ Empieza con la pregunta del entrevistador si la respuesta es el momento potente
✓ Empieza con la frase de setup que hace entendible el remate: "La gente cree que el dinero da felicidad..."
✓ Si el gancho es completamente autoexplicativo ("Gané mi primer millón a los 22"), empieza ahí directamente

INICIO INCORRECTO:
✗ Mitad de frase: el orador está diciendo algo y el clip empieza a mitad de esa frase
✗ Mitad de historia: la historia ya empezó y el clip entra cuando ya hay contexto perdido
✗ Una palabra suelta sin contexto: "...increíble." (¿qué fue increíble?)
✗ El espectador no sabría de qué hablan si empieza a verlo desde ese punto

━━━ LEY DEL CIERRE — EL CLIP DEBE TERMINAR COMPLETO ━━━
El clip termina cuando el orador ha dicho su ÚLTIMA palabra sobre ese tema Y hay un cierre natural.

PRUEBA DEL CIERRE: Lee la última frase del clip. ¿La idea está completa? ¿O el orador iba a decir algo más?
Si iba a decir algo más → EXTIENDE hasta que termine.

CIERRE CORRECTO:
✓ El orador llegó al remate o conclusión de lo que estaba contando
✓ Hay una pausa, risa, o reacción después de la última frase
✓ El entrevistador cambia completamente de tema
✓ La última frase del clip tiene sentido por sí sola como cierre

CIERRE INCORRECTO:
✗ La última frase termina con "y entonces...", "y ahí fue cuando...", "pero lo que nadie sabe es..." → LO IMPORTANTE VIENE DESPUÉS, extiende
✗ El orador acaba de empezar a explicar algo y el clip termina antes de la explicación
✗ El clip termina justo antes del momento más potente
✗ La frase final queda en el aire sin resolución
✗ El entrevistador pregunta algo sobre lo mismo y el orador iba a responder

REGLA DE ORO: Si dudas si extender 15 segundos más → SIEMPRE extiende. Un clip que termina bien siempre gana a uno cortado en seco.

━━━ UN CLIP = UNA IDEA COMPLETA ━━━
Inicio: donde empieza el contexto que hace entendible el momento potente.
Fin: donde termina completamente esa idea, incluyendo remate y reacción breve.
Si el entrevistador cambia de tema → ahí termina el clip.

━━━ EXIGENCIA ━━━
Preferible 2 clips excelentes que 5 mediocres.
Antes de incluir un clip: ¿alguien que no vio el video entendería de qué va y quedaría satisfecho con el final?
Si la respuesta no es SÍ claro → DESCÁRTALO o ajusta los tiempos.

━━━ DURACIÓN — REGLA CRÍTICA ━━━
⚠️ Cada clip DEBE durar entre 30 y 110 segundos. SIN EXCEPCIONES.
- MÍNIMO 30 segundos. Si un momento dura menos de 30s, amplíalo incluyendo contexto antes y después hasta llegar a 30s.
- MÁXIMO 110 segundos. Si dura más de 110s, divídelo en 2 clips más pequeños de 30-60s cada uno.
- EJEMPLO CORRECTO  : "inicio": 269, "fin": 340  →  71 segundos ✓
- EJEMPLO INCORRECTO: "inicio": 269, "fin": 276  →   7 segundos ✗ MUY CORTO
- EJEMPLO INCORRECTO: "inicio":   0, "fin": 626  → 626 segundos ✗ MUY LARGO
- UN SOLO TEMA por clip

━━━ REGLAS ━━━
- Tiempos entre 0 y {duracion}s
- Clips sobre momentos DISTINTOS (sin solapamiento)
- Devuelve TODOS los clips ganadores — sin límite artificial

TRANSCRIPCIÓN:
{transcript}

Responde SOLO con JSON válido, sin texto adicional:
[
  {{
    "tipo": "VERDAD",
    "titulo": "Título máximo 7 palabras que engancha",
    "razon": "En una frase: por qué este clip hace que la gente se quede",
    "segmentos": [{{"inicio": 0.0, "fin": 0.0}}]
  }}
]

TIPOS válidos: VERDAD, REFLEXION, REVELACION, HUMOR, EMOCIONAL, DATO, POLEMICA"""


def _llamar_llama_chunk(prompt_txt):
    """Envía un chunk de transcripción a Llama y devuelve los clips encontrados."""
    for intento in range(3):
        try:
            resp = _groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt_txt}],
                temperature=0.4,
                max_tokens=2000,
            )
            texto = resp.choices[0].message.content.strip()
            m = re.search(r'\[.*\]', texto, re.DOTALL)
            if not m:
                return []
            return json.loads(m.group(0))
        except Exception as e:
            if "413" in str(e):
                print(f"   ⚠️ Chunk demasiado grande (413), saltando")
                return []
            if "429" in str(e):
                print(f"   ⏳ Rate limit Llama, esperando 30s...")
                time.sleep(30)
            else:
                print(f"   ⚠️ Llama: {e}")
                return []
    return []


def _deduplicar_clips(clips, duracion_total):
    """
    Filtra y valida clips:
    1. Duración obligatoria 30-110s (expande cortos con ±15s, descarta largos y no expandibles)
    2. Elimina clips con >50% solapamiento con otro ya aceptado
    """
    DUR_MIN = 30
    DUR_MAX = 110
    EXPAND  = 15

    validos_raw = []
    for c in clips:
        segs = [
            s for s in c.get("segmentos", [])
            if float(s.get("fin", 0)) > float(s.get("inicio", 0))
            and float(s.get("fin", 0)) <= duracion_total + 5
        ]
        if not segs:
            continue

        ini = min(float(s["inicio"]) for s in segs)
        fin = max(float(s["fin"])    for s in segs)
        dur = fin - ini
        titulo = c.get("titulo", "?")

        if dur > DUR_MAX:
            print(f"   ⏭️ Descartado '{titulo}' — {dur:.0f}s > {DUR_MAX}s máximo")
            continue

        if dur < DUR_MIN:
            ini_exp = max(0.0, ini - EXPAND)
            fin_exp = min(duracion_total, fin + EXPAND)
            if fin_exp - ini_exp >= DUR_MIN:
                print(f"   📏 Expandido '{titulo}': {dur:.0f}s → {fin_exp - ini_exp:.0f}s")
                segs = [{"inicio": ini_exp, "fin": fin_exp}]
            else:
                print(f"   ⏭️ Descartado '{titulo}' — {dur:.0f}s < {DUR_MIN}s (no expandible)")
                continue

        c["segmentos"] = segs
        validos_raw.append(c)

    # Eliminar solapamientos >50%
    aceptados = []
    for c in validos_raw:
        ini_c = min(float(s["inicio"]) for s in c["segmentos"])
        fin_c = max(float(s["fin"])    for s in c["segmentos"])
        dur_c = max(fin_c - ini_c, 0.1)

        solapado = False
        for ac in aceptados:
            ini_ac = min(float(s["inicio"]) for s in ac["segmentos"])
            fin_ac = max(float(s["fin"])    for s in ac["segmentos"])
            overlap = max(0.0, min(fin_c, fin_ac) - max(ini_c, ini_ac))
            if overlap / dur_c > 0.5:
                solapado = True
                break

        if not solapado:
            aceptados.append(c)

    return sorted(aceptados, key=lambda c: min(float(s["inicio"]) for s in c["segmentos"]))


def detectar_clips_ia(transcript, duracion_total):
    """
    Llama analiza la transcripción dividida en chunks (evita error 413).
    Cada chunk se procesa independientemente; los clips se fusionan y deduplicán.
    """
    if not _groq or not transcript.strip():
        return []

    # Dividir el transcript en chunks de ~6000 chars (≈1500 tokens)
    # El transcript ya tiene timestamps absolutos por línea ("t=361.5s: texto...")
    CHUNK_CHARS = 6000
    OVERLAP_CHARS = 500  # solapamiento entre chunks para no perder clips en el corte

    lineas = [l for l in transcript.strip().split('\n') if l.strip()]
    chunks = []
    current, current_len = [], 0
    for linea in lineas:
        if current_len + len(linea) > CHUNK_CHARS and current:
            chunks.append('\n'.join(current))
            # solapamiento: empezar el siguiente chunk con las últimas líneas
            overlap_lines = []
            overlap_len = 0
            for ol in reversed(current):
                if overlap_len + len(ol) > OVERLAP_CHARS:
                    break
                overlap_lines.insert(0, ol)
                overlap_len += len(ol)
            current = overlap_lines + [linea]
            current_len = sum(len(l) for l in current)
        else:
            current.append(linea)
            current_len += len(linea)
    if current:
        chunks.append('\n'.join(current))

    print(f"   📋 Transcripción → {len(chunks)} chunk(s) para Llama ({len(lineas)} segmentos)")

    todos_clips = []
    for i, chunk in enumerate(chunks):
        print(f"   🤖 Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        prompt = _PROMPT_CLIPS_IA.format(transcript=chunk, duracion=int(duracion_total))
        clips_chunk = _llamar_llama_chunk(prompt)
        print(f"      → {len(clips_chunk)} clips en chunk {i+1}")
        todos_clips.extend(clips_chunk)

    resultado = _deduplicar_clips(todos_clips, duracion_total)
    print(f"   ✅ {len(resultado)} clips únicos tras deduplicación (de {len(todos_clips)} totales)")
    return resultado
