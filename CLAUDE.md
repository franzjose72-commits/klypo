# KLYPO.IA — Contexto del Proyecto

## Qué es KLYPO
IA que convierte podcasts de YouTube en clips virales para TikTok/Reels de forma 100% automática.
El usuario pega un link de YouTube → KLYPO descarga, transcribe, detecta momentos virales y genera clips verticales 9:16 listos para subir.

## Stack técnico
- Python 3.10, Windows, entorno virtual venv
- `motor.py` — descarga YouTube (yt-dlp + cookies llave.txt), transcribe con Groq Whisper, detecta clips con Llama 3.1
- `nero_editor_viral.py` — edita video (MoviePy), detecta caras (MediaPipe), diariza voces (Pyannote)
- `.env` — variables: GROQ_API_KEY y HF_TOKEN (HuggingFace para Pyannote)

## Versión actual: V46
Última mejora: Split Screen (plano con 2 personas → mitad superior = orador, mitad inferior = entrevistador, resultado 9:16). Fix Pyannote DiarizeOutput named tuple (.annotation → [0] → directo).

## Arquitectura multimodal
- CAPA 1 AUDIO — Pyannote diariza quién habla cada 0.5 segundos
- CAPA 2 VIDEO — MediaPipe detecta caras + OpenCV detecta cortes de cámara
- CAPA 3 FUSIÓN — une audio + video para saber a quién enfocar en cada plano
- Audio y video se procesan INDEPENDIENTEMENTE y se fusionan al final

## Lógica de cámara
- Plano individual (1 cara) → centra en esa cara, fija hasta el siguiente corte
- Plano de orador hablando → enfoca al orador inmediatamente usando historial
- Plano de reacción → enfoca al entrevistador correcto usando diarización
- Cambio de plano → corte directo si supera 25% del ancho
- Dentro del mismo plano → cámara 100% fija, sin temblor
- Speaker estabilizado mínimo 3 segundos (6 intervalos de 0.5s) antes de confirmar cambio

## Problemas resueltos
- Descarga YouTube con cookies y reintentos automáticos
- Rate limit de Groq con espera automática
- Diarización con Pyannote usando scipy (evita error torchcodec)
- Detección de cortes reales de cámara con OpenCV
- Eliminación de clips duplicados
- Solo analiza desde el minuto 5 (evita intro editada)
- Clips únicos sin solapamiento mayor al 50%
- Encuadre 9:16 perfecto con límites verificados
- Sin zoom — solo centrado
- Fix V38: lookahead en cortes de plano + registro continuo de historial de speakers

## Clips objetivo
- Duración: 30 a 110 segundos
- Momentos virales: risas, revelaciones, anécdotas, datos sorprendentes, tensión, humor, momentos WTF
- El nombre del clip debe ser fiel a lo que se dice — estilo TikTok pero congruente

## Roadmap
1. Subtítulos animados estilo MrBeast con identidad KLYPO (próximo)
2. Música de fondo automática
3. Modo Creator — clips cortos de YouTube (fails, highlights, momentos divertidos) para TikTok/Reels/Shorts
4. Split screen automático
5. Open loops — conectar final con inicio para viralidad

## Identidad de marca
- Nombre: KLYPO.IA
- Colores: negro profundo + púrpura ultravioleta (#7C3AED) + cian eléctrico (#06d6f0)
- Tagline: "VIRAL CLIP INTELLIGENCE"

## Archivos clave
- `nero_editor_viral.py` — editor principal (V38 actual)
- `motor.py` — cerebro de descarga y análisis
- `llave.txt` — cookies de YouTube (formato Netscape, 3KB+)
- `.env` — claves API (GROQ_API_KEY, HF_TOKEN)
- `CLIPS_KLYPO_V38/` — carpeta de salida de clips editados

## Forma de trabajar
- Siempre versionar: V38, V39, etc.
- Carpeta de salida cambia con cada versión (CLIPS_KLYPO_V38, CLIPS_KLYPO_V39...)
- Primero diagnosticar el problema, luego aplicar el fix mínimo necesario
- No romper lo que ya funciona
