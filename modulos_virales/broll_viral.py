"""
KLYPO VIRAL — B-roll automático (Pexels + carpeta local mi_broll/)

Flujo por clip:
  1. Llama analiza words_clip y elige 2-5 momentos con su search_term.
  2. Según fuente_broll ("local", "pexels", "ambos"):
       local  → Llama empareja search_term con archivos de mi_broll/
       pexels → busca y descarga de Pexels (igual que antes)
       ambos  → intenta local primero, Pexels como fallback por momento
  3. ffmpeg overlay inserta cada b-roll sobre el video 9:16 ya renderizado.

Si cualquier paso falla el clip original queda intacto — el b-roll es opcional.
"""

import os
import re
import json
import time
import subprocess
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

try:
    _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception:
    _groq = None

_PEXELS_KEY = os.getenv("PEXELS_API_KEY", "")

BROLL_DUR        = 2.5    # segundos que dura cada inserción de b-roll
MIN_GAP_S        = 15.0   # mínimo entre dos b-rolls consecutivos
MAX_BROLLS       = 5      # máximo de b-rolls por clip
MIN_CLIP_T       = 3.0    # no insertar en los primeros N segundos
_TMP_DIR         = "__broll_tmp"
CARPETA_LOCAL    = "mi_broll"
EXTENSIONES_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ─── PROMPTS ─────────────────────────────────────────────────────────────────

_PROMPT_BROLL = """Eres un EDITOR DE VIDEO VIRAL profesional. Tu trabajo es elegir b-roll (imágenes de apoyo) que refuercen el MENSAJE de lo que dice el orador, como lo haría un buen editor.

NO busques la palabra literal. Interpreta QUÉ QUIERE DECIR y elige una escena que lo represente con sentido.

Ejemplos de cómo piensa un editor:
- "Me junté con personas que están donde yo quiero estar" → NO es "grupo de personas". Es gente exitosa, ambiente de negocios, lujo, networking de alto nivel. Search: "successful business people meeting luxury"
- "Trabajé duro toda mi vida" → escena de esfuerzo: alguien trabajando de noche, sudando, dedicación. Search: "person working late night dedication"
- "El dinero no lo es todo" → contraste: lujo vacío, soledad con dinero. Search: "lonely rich person luxury empty"
- "Cuando toqué fondo" → escena de caída, oscuridad, momento bajo. Search: "person sad alone dark moment"
- "Me compré un Lamborghini" → no el coche vacío: velocidad, lujo en movimiento. Search: "lamborghini driving speed luxury"
- "Dejé mi trabajo de oficina" → oficina gris, rutina sofocante, salir por la puerta. Search: "person leaving office job freedom"

EVITA imágenes literales y vacías de conceptos abstractos:
- "Todos somos hijos de Dios" → NO el cielo. SÍ: gente unida, alguien ayudando a otro, comunidad, manos juntas. Search: "people helping each other community"
- "La fe te salva" → NO una cruz flotando. SÍ: alguien superando una dificultad, levantándose tras caer. Search: "person overcoming struggle hope"
- "El propósito de tu vida" → NO una luz abstracta. SÍ: alguien trabajando con pasión en lo suyo. Search: "person passionate about their work purpose"

Piensa SIEMPRE en una escena con PERSONAS haciendo algo concreto que represente el mensaje. Nunca conceptos flotando o paisajes vacíos sin acción humana.

REGLAS:
- Interpreta el contexto y la emoción, no la palabra suelta
- El b-roll debe hacer que el espectador SIENTA o ENTIENDA mejor el mensaje
- Elige entre 2 y 5 momentos, mínimo 15 segundos entre cada uno
- Solo momentos donde una imagen REFUERCE de verdad lo que se dice
- Si un momento no tiene una imagen clara que lo mejore, NO lo incluyas
- Los términos de búsqueda (search_term) SIEMPRE en inglés y descriptivos de la ESCENA, no de la palabra literal
- No en los primeros 3 segundos del clip (el gancho no se interrumpe)
- `t` es el segundo exacto en el clip (relativo, empieza en 0)

TRANSCRIPCIÓN CON TIMESTAMPS:
{transcript}

Duración del clip: {duracion}s

Responde SOLO con JSON válido, sin texto adicional:
[
  {{"t": 8.5, "keyword": "frase clave que motiva el b-roll", "search_term": "descriptive english scene"}},
  {{"t": 28.0, "keyword": "me junté con los mejores", "search_term": "successful business people meeting luxury"}}
]

Si no hay momentos donde una imagen mejore claramente el mensaje, devuelve [].
"""

_PROMPT_MATCH_LOCAL = """Tienes una lista de videos de b-roll disponibles y una lista de momentos que necesitan b-roll.
Tu trabajo: para cada search_term, elige qué archivo encaja mejor semánticamente.
Los archivos pueden estar nombrados en español y los search_terms están en inglés — haz el match por significado, no por idioma.

ARCHIVOS DISPONIBLES (nombre sin extensión):
{archivos}

SEARCH TERMS A EMPAREJAR:
{momentos}

Para cada search_term devuelve el nombre exacto del archivo que mejor encaja, o null si ninguno encaja razonablemente.

Responde SOLO con JSON válido, sin texto adicional:
{{"successful business people meeting luxury": "negocios exito empresa", "person working out gym": "gym entrenamiento", "sad dark moment": null}}
"""


# ─── 1. DETECCIÓN DE MOMENTOS ─────────────────────────────────────────────────

def _hay_palabras_en(t, words_clip, ventana=0.5):
    """True si el momento t cae dentro de una palabra hablada (±0.5s de margen)."""
    for w in words_clip:
        if w["start"] - ventana <= t <= w["end"] + ventana:
            return True
    return False


def detectar_momentos_broll(words_clip, duracion_clip):
    """
    Usa Llama para elegir momentos de b-roll a partir de words_clip ya remapeados.
    words_clip: [{word, start, end}] con timestamps relativos al clip ensamblado.
    Devuelve [{t, keyword, search_term}] o [] si no hay nada válido.
    """
    if not words_clip or not _groq:
        return []

    lineas = [f"t={w['start']:.1f}s: {w['word']}" for w in words_clip]
    transcript = "\n".join(lineas)

    prompt = _PROMPT_BROLL.format(
        transcript=transcript[:4000],
        duracion=int(duracion_clip),
    )

    try:
        resp = _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        texto = resp.choices[0].message.content.strip()
        m = re.search(r'\[.*\]', texto, re.DOTALL)
        if not m:
            print("   🎥 B-roll: Llama no devolvió JSON válido")
            return []

        candidatos = json.loads(m.group(0))
        validos = []
        ultimo_t = -MIN_GAP_S

        for mo in candidatos:
            t           = float(mo.get("t", 0))
            keyword     = str(mo.get("keyword", "")).strip()
            search_term = str(mo.get("search_term", "")).strip()

            if not keyword or not search_term:
                continue
            if t < MIN_CLIP_T:
                print(f"   🎥 B-roll: '{keyword}' en t={t:.1f}s saltado — demasiado al inicio")
                continue
            if not _hay_palabras_en(t, words_clip):
                print(f"   🎥 B-roll: '{keyword}' en t={t:.1f}s saltado — silencio")
                continue
            if t + BROLL_DUR > duracion_clip:
                continue
            if t - ultimo_t < MIN_GAP_S:
                print(f"   🎥 B-roll: '{keyword}' en t={t:.1f}s saltado — muy cerca del anterior")
                continue

            validos.append({"t": t, "keyword": keyword, "search_term": search_term})
            ultimo_t = t

            if len(validos) >= MAX_BROLLS:
                break

        print(f"   🎥 B-roll: {len(validos)} momento(s) elegidos por Llama")
        return validos

    except Exception as e:
        print(f"   ⚠️ B-roll detección: {e}")
        return []


# ─── 2. CARPETA LOCAL ─────────────────────────────────────────────────────────

def listar_videos_locales():
    """
    Escanea mi_broll/ en la raíz del proyecto.
    Devuelve lista de rutas absolutas de archivos de video válidos.
    """
    raiz    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carpeta = os.path.join(raiz, CARPETA_LOCAL)

    if not os.path.isdir(carpeta):
        print(f"   📁 B-roll local: carpeta '{CARPETA_LOCAL}/' no existe — saltando videos locales")
        return []

    archivos = [
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if os.path.splitext(f)[1].lower() in EXTENSIONES_VIDEO
    ]

    if not archivos:
        print(f"   📁 B-roll local: '{CARPETA_LOCAL}/' está vacía — saltando videos locales")
    else:
        print(f"   📁 B-roll local: {len(archivos)} video(s) en '{CARPETA_LOCAL}/'")

    return archivos


def buscar_matches_locales(momentos, archivos_locales):
    """
    Una sola llamada a Llama para emparejar todos los search_terms con los archivos locales.
    Devuelve {search_term -> ruta_absoluta | None}.
    Si Llama falla devuelve {} (todos los momentos quedarán sin match local).
    """
    if not _groq or not archivos_locales or not momentos:
        return {}

    # Nombre legible: sin extensión, guiones bajos → espacios
    nombre_a_ruta = {}
    for ruta in archivos_locales:
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        nombre_legible = nombre.replace("_", " ")
        nombre_a_ruta[nombre_legible] = ruta

    archivos_str = "\n".join(f"- {n}" for n in nombre_a_ruta)
    momentos_str = "\n".join(f"- \"{mo['search_term']}\"" for mo in momentos)

    prompt = _PROMPT_MATCH_LOCAL.format(
        archivos=archivos_str,
        momentos=momentos_str,
    )

    try:
        resp = _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        texto = resp.choices[0].message.content.strip()
        m = re.search(r'\{.*\}', texto, re.DOTALL)
        if not m:
            print("   ⚠️ B-roll local: Llama no devolvió JSON de matching")
            return {}

        raw = json.loads(m.group(0))

        resultado = {}
        for search_term, nombre_match in raw.items():
            if not nombre_match:
                resultado[search_term] = None
                continue
            ruta = nombre_a_ruta.get(str(nombre_match))
            if ruta and os.path.exists(ruta) and os.path.getsize(ruta) > 1000:
                resultado[search_term] = ruta
            else:
                resultado[search_term] = None

        return resultado

    except Exception as e:
        print(f"   ⚠️ B-roll local matching: {e}")
        return {}


# ─── 3. PEXELS ───────────────────────────────────────────────────────────────

def buscar_video_pexels(search_term):
    """
    Busca en Pexels un video corto (4-20s) para el término dado.
    Devuelve la URL de descarga HD o None.
    """
    if not _PEXELS_KEY:
        print("   ⚠️ B-roll: PEXELS_API_KEY no está en .env")
        return None

    try:
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": _PEXELS_KEY},
            params={"query": search_term, "per_page": 10, "orientation": "landscape"},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"   ⚠️ B-roll Pexels: status {resp.status_code} para '{search_term}'")
            return None

        videos = resp.json().get("videos", [])
        if not videos:
            print(f"   ⚠️ B-roll Pexels: sin resultados para '{search_term}'")
            return None

        videos_cortos = sorted(
            [v for v in videos if 4 <= v.get("duration", 999) <= 20],
            key=lambda v: v.get("duration", 999),
        )
        candidatos = videos_cortos if videos_cortos else videos[:3]

        for video in candidatos:
            archivos = video.get("video_files", [])
            hd = [f for f in archivos if f.get("quality") == "hd" and f.get("width", 0) <= 1920]
            if hd:
                mejor = max(hd, key=lambda f: f.get("width", 0))
                if mejor.get("link"):
                    return mejor["link"]
            for f in sorted(archivos, key=lambda f: f.get("width", 0), reverse=True):
                if f.get("link"):
                    return f["link"]

        return None

    except Exception as e:
        print(f"   ⚠️ B-roll Pexels búsqueda: {e}")
        return None


def descargar_clip_pexels(url, output_path):
    """Descarga un clip de Pexels. Devuelve True si fue exitoso."""
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code != 200:
            return False
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        print(f"   ⚠️ B-roll descarga: {e}")
        return False


# ─── 4. OVERLAY CON FFMPEG ───────────────────────────────────────────────────

def aplicar_broll(video_path, momentos_con_clips, output_path):
    """
    Superpone los b-rolls sobre el video 9:16 usando ffmpeg filter_complex.
    momentos_con_clips: [{t, keyword, clip_path}]
    Si falla, el video original en output_path queda intacto.
    """
    validos = [
        (m["t"], m["keyword"], m["clip_path"])
        for m in momentos_con_clips
        if os.path.exists(m.get("clip_path", ""))
    ]
    if not validos:
        print("   🎥 B-roll: ningún clip disponible para overlay")
        return

    tmp_out = video_path.replace(".mp4", "_broll_tmp.mp4")

    inputs = ["-i", video_path]
    for _, _, clip_path in validos:
        inputs += ["-i", clip_path]

    filter_parts = []
    for idx, (t, _, _) in enumerate(validos):
        filter_parts.append(
            f"[{idx + 1}:v]"
            f"scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
            f"setpts=PTS-STARTPTS"
            f"[b{idx}]"
        )

    prev = "0:v"
    for idx, (t, _, _) in enumerate(validos):
        t_end  = round(t + BROLL_DUR, 3)
        salida = f"v{idx}" if idx < len(validos) - 1 else "vfinal"
        filter_parts.append(
            f"[{prev}][b{idx}]"
            f"overlay=0:0:enable='between(t,{t:.3f},{t_end:.3f})'"
            f"[{salida}]"
        )
        prev = salida

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + ["-filter_complex", ";".join(filter_parts)]
        + ["-map", "[vfinal]", "-map", "0:a"]
        + ["-c:v", "libx264", "-crf", "20", "-preset", "fast"]
        + ["-c:a", "copy", tmp_out]
    )

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        if result is None:
            print("   ⚠️ B-roll ffmpeg no se pudo ejecutar — clip original conservado")
            return
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="ignore")
            print(f"   ⚠️ B-roll ffmpeg falló — clip original conservado")
            print(f"      {err[-300:]}")
            if os.path.exists(tmp_out):
                try: os.remove(tmp_out)
                except: pass
            return

        for _i in range(3):
            try:
                os.replace(tmp_out, output_path)
                break
            except OSError:
                time.sleep(0.5)

        kws = ", ".join(kw for _, kw, _ in validos)
        print(f"   🎥 B-roll aplicado: {len(validos)} inserciones ({kws})")

    except subprocess.TimeoutExpired:
        print("   ⚠️ B-roll: ffmpeg timeout — clip original conservado")
        if os.path.exists(tmp_out):
            try: os.remove(tmp_out)
            except: pass
    except Exception as e:
        print(f"   ⚠️ B-roll overlay: {e}")
        if os.path.exists(tmp_out):
            try: os.remove(tmp_out)
            except: pass


# ─── 5. PIPELINE COMPLETO ────────────────────────────────────────────────────

def procesar_broll(words_clip, duracion_clip, output_path, fuente_broll="pexels"):
    """
    Orquesta el pipeline completo de b-roll para un clip ya renderizado.
    fuente_broll: "pexels" | "local" | "ambos"
    Si cualquier paso falla, output_path original queda intacto.
    """
    # Si solo es local, verificar que haya videos ANTES de llamar a Llama
    archivos_locales = []
    if fuente_broll in ("local", "ambos"):
        archivos_locales = listar_videos_locales()
        if not archivos_locales and fuente_broll == "local":
            print("   ℹ️ Sin B-roll disponible (mi_broll/ vacía o inexistente)")
            return

    momentos = detectar_momentos_broll(words_clip, duracion_clip)
    if not momentos:
        return

    # Pre-calcular matches locales en una sola llamada a Llama (si aplica)
    matches_locales = {}
    if fuente_broll in ("local", "ambos") and archivos_locales:
        matches_locales = buscar_matches_locales(momentos, archivos_locales)

    os.makedirs(_TMP_DIR, exist_ok=True)
    momentos_con_clips = []
    tmp_descargados    = []  # solo rutas de Pexels descargadas, para limpiar al final

    for mo in momentos:
        keyword     = mo["keyword"]
        search_term = mo["search_term"]
        t           = mo["t"]
        clip_path   = None

        # Intento 1: video local
        if fuente_broll in ("local", "ambos"):
            ruta_local = matches_locales.get(search_term)
            if ruta_local:
                print(f"   📁 B-roll local: '{keyword}' → {os.path.basename(ruta_local)}")
                clip_path = ruta_local
            elif fuente_broll == "local":
                print(f"   ⚠️ B-roll: sin match local para '{keyword}', saltando")

        # Intento 2: Pexels (si no hay local y aplica)
        if clip_path is None and fuente_broll in ("pexels", "ambos"):
            print(f"   🔍 B-roll Pexels: '{search_term}' (t={t:.1f}s)...")
            url = buscar_video_pexels(search_term)
            if url:
                slug     = re.sub(r'\W+', '_', search_term)[:30]
                tmp_path = os.path.join(_TMP_DIR, f"broll_{slug}.mp4")
                ok       = descargar_clip_pexels(url, tmp_path)
                if ok:
                    print(f"   ✅ B-roll Pexels descargado: '{keyword}'")
                    clip_path = tmp_path
                    tmp_descargados.append(tmp_path)
                else:
                    print(f"   ⚠️ B-roll: descarga fallida para '{search_term}', saltando")

        if clip_path is None:
            continue

        momentos_con_clips.append({**mo, "clip_path": clip_path})

    if momentos_con_clips:
        aplicar_broll(output_path, momentos_con_clips, output_path)

    # Limpiar solo los archivos descargados de Pexels (nunca los locales)
    for p in tmp_descargados:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    try:
        os.rmdir(_TMP_DIR)
    except Exception:
        pass
