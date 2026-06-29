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

def transcribir_segmento(archivo_segmento):
    # Intento 1: Groq Whisper — si hay rate limit, cae inmediatamente a OpenAI (sin esperar)
    try:
        print(f"   🎙️ Transcribiendo con Groq Whisper...")
        with open(archivo_segmento, "rb") as file:
            return client_groq.audio.transcriptions.create(
                file=(archivo_segmento, file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
    except Exception as e:
        if '429' in str(e):
            print(f"⚡ Rate limit Groq → OpenAI Whisper inmediato")
        else:
            print(f"⚠️ Groq Whisper falló: {e} — usando OpenAI como fallback")

    # Fallback: OpenAI Whisper
    for _ in range(3):
        try:
            print(f"   🎙️ Transcribiendo con OpenAI Whisper (fallback)...")
            with open(archivo_segmento, "rb") as f:
                response = client_openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text"
                )
            return response
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                print(f"⏳ Rate limit OpenAI, esperando 30s...")
                time.sleep(30)
            else:
                print(f"⚠️ Error OpenAI Whisper: {e}")
                return ""
    return ""

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

def procesar_segmentos_paralelo(segmentos, duracion_total, max_workers=1):
    """
    Transcribe + detecta ganchos. Con max_workers=1 (secuencial) el rate limit de Groq
    se recupera entre segmentos — evita que todos fallen al mismo tiempo.
    """
    def _procesar_uno(item):
        start, audio_path = item
        print(f"   📝 Procesando segmento {start}s...")
        texto = transcribir_segmento(audio_path)
        if not texto:
            print(f"   ⚠️ Segmento {start}s: transcripción vacía, saltando.")
            return (start, "[]")
        ganchos = buscar_ganchos_en_segmento(texto, start, duracion_total - 5)
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
Eres KLYPO, el mejor editor viral de podcasts para TikTok/Reels. Tu trabajo es extraer momentos que hagan que alguien PARE el scroll.

MOMENTOS QUE BUSCAS:
- Revelaciones impactantes, confesiones, secretos
- Datos contraintuitivos o sorprendentes
- Humor, anécdotas graciosas, momentos WTF
- Opiniones polémicas o controversiales
- Historias personales poderosas o emotivas
- Momentos de tensión o conflicto
- Frases memorables que golpean fuerte

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEY #1 — EL GANCHO ES LO PRIMERO (REGLA MÁS IMPORTANTE)
Los primeros 3 segundos del clip deciden si el espectador sigue o no.
El clip DEBE empezar en la frase MÁS IMPACTANTE disponible.

  TIPOS DE INICIO QUE PARAN EL SCROLL:
  → Pregunta provocadora: "¿Sabes por qué el 90% de la gente fracasa?"
  → Afirmación chocante: "Yo perdí un millón de euros en una semana."
  → Dato que rompe la lógica: "Trabajar más horas te hace menos productivo."
  → Confesión directa: "Nunca te conté esto, pero estuve a punto de dejarlo."
  → Tensión inmediata: "Ese día casi me matan."

  PROHIBIDO EMPEZAR CON:
  ✗ Contexto o relleno: "Bueno, la cosa es que...", "O sea, lo que yo creo..."
  ✗ Continuación de idea anterior: "...y entonces", "...pero lo que pasa"
  ✗ Respuestas a preguntas no escuchadas: "Sí, exacto, lo que dices es..."
  ✗ Saludos, agradecimientos, transiciones: "Claro, claro", "Exactamente"

  Si la parte más impactante de una historia NO está al inicio, empieza
  directamente en esa frase impactante aunque sea del medio de la historia.
  El espectador prefiere entrar en tensión y quedarse, que empezar en frío.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEY #2 — UN CLIP = UN SOLO TEMA (SIN EXCEPCIONES)
El clip trata exactamente UNA idea, UNA historia, UNA pregunta.

  SEÑALES DE QUE EL TEMA CAMBIÓ → CORTAR AHÍ:
  ✗ El entrevistador hace una nueva pregunta sobre un tema diferente
  ✗ El orador dice "Bueno, cambiando de tema...", "Hablando de otra cosa..."
  ✗ Aparece un nuevo nombre, proyecto o concepto no relacionado con el tema del clip
  ✗ El tono cambia de emocional/serio a distendido, o viceversa, por un tema nuevo
  ✗ El orador concluye con "...y ya" o hace una pausa larga antes de empezar algo nuevo

  El clip termina en el remate del tema: afirmación rotunda, risa de cierre,
  moraleja, dato final. NUNCA incluyas el inicio del siguiente tema aunque
  sean solo 5 segundos — eso rompe el foco y el algoritmo lo penaliza.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESTRUCTURA INTERNA DEL CLIP:
  [GANCHO] → primeros 3-8s: la frase que para el scroll (ver Ley #1)
  [DESARROLLO] → al menos 15s: explicación, historia, argumentos
  [REMATE] → cierre con punch: dato final, confesión, moraleja, reacción

DURACIÓN: mínimo 30 segundos, máximo 110 segundos. NUNCA más ni menos.
Los tiempos son RELATIVOS al inicio del segmento (empiezan en 0).
NUNCA repitas el mismo rango de tiempo.

CANTIDAD: entre 3 y 5 clips por segmento. Prioriza calidad sobre cantidad.
Si un segmento tiene solo 2 momentos completos, devuelve 2. NO rellenes con clips mediocres.

TÍTULOS — REGLA FUNDAMENTAL:
El título DEBE referenciar algo CONCRETO que se dice en el clip: un dato, una frase, una anécdota, una cifra, un nombre, una confesión real.
PROHIBIDO: títulos genéricos tipo "Lo que nadie te dice" o "La mayoría lo hace MAL".
OBLIGATORIO: alguien que vea el clip debe reconocer que el título describe exactamente lo que escuchó.
Puede ser provocador, pero SIEMPRE fiel al contenido. Máximo 60 caracteres.

Responde ÚNICAMENTE con JSON válido, sin texto extra:
[
  {"inicio": 0, "fin": 85, "titulo": "Título viral aquí"},
  {"inicio": 120, "fin": 210, "titulo": "Otro momento que engancha"}
]
"""
    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": f"Segmento del segundo {offset_tiempo}:\n\n{transcripcion}"}
                ],
                model="llama-3.3-70b-versatile"
            )
            return completion.choices[0].message.content
        except Exception as e:
            if '429' in str(e):
                print(f"⏳ Rate limit Llama, esperando 20s...")
                time.sleep(20)
            else:
                print(f"⚠️ Error buscando ganchos: {e}")
                return "[]"
    return "[]"
