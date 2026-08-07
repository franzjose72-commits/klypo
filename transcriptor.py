import os
import time
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Clientes de transcripción
client_groq   = Groq(api_key=os.getenv("GROQ_API_KEY"))
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Groq sigue siendo el cliente para análisis de texto (Llama)
client = client_groq

# Sentinels que retorna buscar_ganchos_en_segmento para que el caller distinga:
#   RATE_LIMIT  → cuota por minuto agotada, reintentar en 2-3 min
#   DAILY_LIMIT → cuota horaria/diaria agotada (Retry-After > 90s), reintentar en horas
GROQ_RATE_LIMIT_SENTINEL  = "__GROQ_RATE_LIMIT__"
GROQ_DAILY_LIMIT_SENTINEL = "__GROQ_DAILY_LIMIT__"

# Si Retry-After supera este umbral se trata como límite horario/diario → abortar de inmediato
_RETRY_AFTER_ABORT = 90

def transcribir_segmento(archivo_segmento):
    """Devuelve (texto_con_timestamps, segs) donde segs=[{"start":float,"end":float}].
    El texto tiene el formato '[t=Xs-Xs] oración' para que el modelo use timestamps exactos."""

    def _parsear_verbose(resp):
        segs_raw = (resp.get("segments") if isinstance(resp, dict)
                    else getattr(resp, "segments", None)) or []
        segs, lineas = [], []
        for s in segs_raw:
            if isinstance(s, dict):
                t0  = float(s.get("start", 0))
                t1  = float(s.get("end", 0))
                txt = s.get("text", "").strip()
            else:
                t0  = float(s.start)
                t1  = float(s.end)
                txt = s.text.strip()
            if txt:
                lineas.append(f"[{int(t0)}s] {txt}")
                segs.append({"start": t0, "end": t1})
        return "\n".join(lineas), segs

    # Intento 1: Groq Whisper con timestamps de segmento
    try:
        print(f"   🎙️ Transcribiendo con Groq Whisper...")
        with open(archivo_segmento, "rb") as file:
            resp = client_groq.audio.transcriptions.create(
                file=(archivo_segmento, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        texto, segs = _parsear_verbose(resp)
        if texto:
            return texto, segs
    except Exception as e:
        if '429' in str(e):
            print(f"⚡ Rate limit Groq → OpenAI Whisper inmediato")
        else:
            print(f"⚠️ Groq Whisper falló: {e} — usando OpenAI como fallback")

    # Fallback: OpenAI Whisper con timestamps de segmento
    for _ in range(3):
        try:
            print(f"   🎙️ Transcribiendo con OpenAI Whisper (fallback)...")
            with open(archivo_segmento, "rb") as f:
                response = client_openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            texto, segs = _parsear_verbose(response)
            return texto, segs
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                print(f"⏳ Rate limit OpenAI, esperando 30s...")
                time.sleep(30)
            else:
                print(f"⚠️ Error OpenAI Whisper: {e}")
                return "", []
    return "", []

def transcribir_clip_timestamps(audio_path):
    # Intento 1: Groq Whisper con timestamps — si hay rate limit, OpenAI inmediato
    try:
        print(f"   🎙️ Timestamps con Groq Whisper...")
        with open(audio_path, "rb") as f:
            resp = client_groq.audio.transcriptions.create(
                file=(audio_path, f.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
        words_data = (resp.get("words") if isinstance(resp, dict)
                      else getattr(resp, "words", None)) or []
        if words_data:
            result = []
            for w in words_data:
                if isinstance(w, dict):
                    texto = w.get("word", "").strip()
                    start = float(w.get("start", 0))
                    end   = float(w.get("end", 0))
                else:
                    texto = w.word.strip()
                    start = float(w.start)
                    end   = float(w.end)
                if texto:
                    result.append({"word": texto, "start": start, "end": end})
            return result
        return []
    except Exception as e:
        if '429' in str(e):
            print(f"⚡ Rate limit Groq → OpenAI Whisper timestamps inmediato")
        else:
            print(f"⚠️ Groq Whisper timestamps falló: {e} — usando OpenAI como fallback")

    # Fallback: OpenAI Whisper con timestamps
    for _ in range(3):
        try:
            print(f"   🎙️ Timestamps con OpenAI Whisper (fallback)...")
            with open(audio_path, "rb") as f:
                response = client_openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )
            words_data = getattr(response, "words", None) or []
            if words_data:
                result = []
                for w in words_data:
                    if isinstance(w, dict):
                        texto = w.get("word", "").strip()
                        start = float(w.get("start", 0))
                        end   = float(w.get("end", 0))
                    else:
                        texto = getattr(w, "word", "").strip()
                        start = float(getattr(w, "start", 0))
                        end   = float(getattr(w, "end", 0))
                    if texto:
                        result.append({"word": texto, "start": start, "end": end})
                return result
            return []
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                print(f"⏳ Rate limit OpenAI, esperando 30s...")
                time.sleep(30)
            else:
                print(f"⚠️ Error OpenAI timestamps: {e}")
                return []
    return []

_ENCAJE_MAX_MOV = 15   # máx segundos que puede moverse un corte al encajar

def _encajar_clips_json(ganchos_txt, segs):
    """Encaja inicio/fin de cada clip al límite de segmento Whisper más cercano.
    Solo aplica si el movimiento es ≤15s y el resultado cumple 30≤dur≤110s."""
    import re, json as _json
    if not segs or ganchos_txt in (GROQ_RATE_LIMIT_SENTINEL, GROQ_DAILY_LIMIT_SENTINEL):
        return ganchos_txt
    match = re.search(r'\[.*\]', ganchos_txt, re.DOTALL)
    if not match:
        return ganchos_txt
    try:
        clips = _json.loads(match.group())
    except Exception:
        return ganchos_txt
    if not clips:
        return ganchos_txt

    def _snap_ini(t):
        best = t
        for seg in segs:
            if seg['start'] <= t:
                best = seg['start']
            else:
                break
        return best if abs(best - t) <= _ENCAJE_MAX_MOV else t

    def _snap_fin(t):
        for seg in segs:
            if seg['end'] >= t:
                return seg['end'] if abs(seg['end'] - t) <= _ENCAJE_MAX_MOV else t
        last = segs[-1]['end']
        return last if abs(last - t) <= _ENCAJE_MAX_MOV else t

    encajados = []
    for idx, clip in enumerate(clips):
        try:
            ini = float(clip['inicio'])
            fin = float(clip['fin'])
            ini_new = _snap_ini(ini)
            fin_new = _snap_fin(fin)
            dur_new = fin_new - ini_new
            if 30 <= dur_new <= 60:
                if abs(ini_new - ini) > 0.5 or abs(fin_new - fin) > 0.5:
                    print(f"   📐 Encaje clip {idx+1}: {ini:.1f}s-{fin:.1f}s → {ini_new:.1f}s-{fin_new:.1f}s")
                clip = {**clip, 'inicio': round(ini_new, 1), 'fin': round(fin_new, 1)}
        except Exception:
            pass
        encajados.append(clip)

    return _json.dumps(encajados, ensure_ascii=False)


def procesar_segmentos_paralelo(segmentos, duracion_total, max_workers=1):
    """
    Transcribe + detecta ganchos. Con max_workers=1 (secuencial) el rate limit de Groq
    se recupera entre segmentos — evita que todos fallen al mismo tiempo.
    """
    def _procesar_uno(item):
        start, audio_path = item
        print(f"   📝 Procesando segmento {start}s...")
        texto, segs = transcribir_segmento(audio_path)
        if not texto:
            print(f"   ⚠️ Segmento {start}s: transcripción vacía, saltando.")
            return (start, "[]")
        ganchos = buscar_ganchos_en_segmento(texto, start, duracion_total - 5)
        ganchos = _encajar_clips_json(ganchos, segs)
        print(f"   ✅ Segmento {start}s listo.")
        return (start, ganchos)

    resultados = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_procesar_uno, seg): seg for seg in segmentos}
        for fut in as_completed(futures):
            try:
                resultados.append(fut.result())
            except Exception as e:
                start = futures[fut][0]
                print(f"⚠️ Error paralelo segmento {start}s: {e}")
                resultados.append((start, "[]"))
    return sorted(resultados, key=lambda x: x[0])

def buscar_ganchos_en_segmento(transcripcion, offset_tiempo, _duracion_maxima=None):
    prompt_sistema = """
Eres KLYPO, editor viral de podcasts para TikTok/Reels. Extrae momentos que paren el scroll: revelaciones, datos contraintuitivos, humor WTF, opiniones polémicas, historias emotivas, tensión, frases memorables.

LEY 1 — GANCHO PRIMERO (regla más importante)
Los primeros 3s deciden si el espectador sigue. El clip DEBE empezar en la frase más impactante, aunque esté en el medio de la historia.
Inicios válidos: pregunta provocadora, afirmación chocante, dato que rompe la lógica, confesión directa, tensión inmediata.
PROHIBIDO empezar: contexto/relleno ("Bueno, la cosa es..."), continuación de idea anterior, respuesta a pregunta no escuchada, saludos o transiciones.

LEY 2 — UN CLIP = UN SOLO TEMA
Cuando el orador cambia de tema, cortar ahí. El clip trata exactamente UNA idea de inicio a fin.
El clip termina en el remate: afirmación rotunda, moraleja, dato final. NUNCA incluyas el inicio del siguiente tema.

ESTRUCTURA: [GANCHO 3-8s] — [DESARROLLO min 15s] — [REMATE con punch]
DURACIÓN: 30-60s exactos. El texto tiene marcas [Xs] con el segundo exacto de inicio de cada oración — usa SOLO esos valores como 'inicio', elige el [Xs] de la última oración del clip como 'fin'. No estimes. NUNCA repitas el mismo rango.
CANTIDAD: 3-5 clips. Si solo hay 2 momentos completos, devuelve 2. No rellenes con clips mediocres.
TÍTULOS: referencia algo CONCRETO del clip (dato, cifra, frase real, anécdota). PROHIBIDO títulos genéricos. Max 60 chars.

Responde ÚNICAMENTE con JSON válido, sin texto extra:
[{"inicio": 0, "fin": 85, "titulo": "Título viral aquí"}]
"""
    for intento in range(1, 6):   # 5 intentos máximo
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": f"Segmento del segundo {offset_tiempo}:\n\n{transcripcion}"}
                ],
                model="llama-3.1-8b-instant",
                max_tokens=1000,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if '429' in str(e):
                espera = 60  # default: 1 ventana de rate-limit por minuto
                try:
                    resp = getattr(e, 'response', None)
                    hdrs = dict(getattr(resp, 'headers', {}) or {})
                    ra   = hdrs.get('retry-after') or hdrs.get('Retry-After') or ''
                    if ra:
                        if 'm' in ra:  # formato "1m30s"
                            m, s = ra.lower().replace('s', '').split('m')
                            espera = int(float(m)) * 60 + int(float(s or '0'))
                        else:
                            espera = int(float(ra))
                except Exception:
                    pass

                if espera > _RETRY_AFTER_ABORT:
                    # Límite horario o diario — esperar sería inutilizable para el usuario
                    print(
                        f"❌ Groq: límite horario/diario alcanzado "
                        f"(Retry-After={espera}s). Abortando sin esperar."
                    )
                    return GROQ_DAILY_LIMIT_SENTINEL

                espera = max(espera, 60)  # nunca menos de 60s para límites por minuto
                print(f"⏳ Rate limit Llama (intento {intento}/5), esperando {espera}s...")
                if intento < 5:
                    time.sleep(espera)
            else:
                print(f"⚠️ Error buscando ganchos: {e}")
                return "[]"
    print(f"❌ Rate limit Groq agotado tras 5 intentos — segmento {offset_tiempo}s sin analizar")
    return GROQ_RATE_LIMIT_SENTINEL
