"""
KLYPO - Subida de clips a Cloudflare R2

Variables requeridas en RunPod:
  R2_ACCESS_KEY_ID     - Access Key ID del token API de R2
  R2_SECRET_ACCESS_KEY - Secret Access Key del token API de R2
  R2_ENDPOINT          - https://<ACCOUNT_ID>.r2.cloudflarestorage.com
  R2_BUCKET            - nombre del bucket (ej: klypo-clips)

Estrategia de subida — doble proteccion contra proxy:
  1. boto3 genera la presigned PUT URL LOCALMENTE (firma criptografica, sin red)
  2. requests.put() sube el archivo con proxies={"http": None, "https": None}
     None explicito ignora HTTP_PROXY/HTTPS_PROXY aunque esten en el entorno.
     Esto es diferente a "" (cadena vacia) que algunos clientes HTTP ignoran.
"""

import os
import glob
import time
import traceback

import boto3.session as b3s
import certifi
import requests
from botocore.config import Config


# Todas las variables de entorno que pueden desviar trafico HTTP/HTTPS por un proxy
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
               "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")


# ── Diagnostico y gestion de proxy ───────────────────────────────────────────

def _log_proxy_estado(tag: str):
    """Imprime el estado real de cada variable de proxy en os.environ."""
    presentes = {v: os.environ[v] for v in _PROXY_VARS if v in os.environ}
    if presentes:
        print(f"   ⚠️  [{tag}] Proxy vars ACTIVAS en entorno:")
        for k, v in presentes.items():
            print(f"         {k} = {v!r}")
    else:
        print(f"   ✅ [{tag}] Sin variables de proxy en el entorno")


def _limpiar_proxy() -> dict:
    """
    Elimina todas las vars de proxy e inyecta NO_PROXY=*.
    Devuelve el estado original para restaurar despues.
    """
    guardadas = {v: os.environ.pop(v) for v in _PROXY_VARS if v in os.environ}
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    return guardadas


def _restaurar_proxy(guardadas: dict):
    """Elimina los NO_PROXY que inyectamos y restaura el estado original."""
    os.environ.pop("NO_PROXY", None)
    os.environ.pop("no_proxy", None)
    os.environ.update(guardadas)


# ── Utilidades de archivo ─────────────────────────────────────────────────────

def _encontrar_archivo(ruta_esperada: str) -> "str | None":
    """
    Devuelve la ruta real del archivo aunque el nombre difiera levemente.
    Primero la ruta exacta; si no existe, busca por prefijo de 30 chars.
    """
    if os.path.exists(ruta_esperada):
        return ruta_esperada

    directorio = os.path.dirname(ruta_esperada)
    nombre     = os.path.basename(ruta_esperada)
    patron     = os.path.join(directorio, nombre[:30] + "*.mp4")
    candidatos = glob.glob(patron)
    if candidatos:
        encontrado = candidatos[0]
        print(f"   ⚠️  Ruta exacta no hallada, usando: {os.path.basename(encontrado)}")
        return encontrado

    print(f"   ❌ Archivo no encontrado en disco: {ruta_esperada}")
    return None


# ── Cliente boto3 (solo para firmar URLs localmente) ─────────────────────────

def _crear_cliente_s3(access_key: str, secret_key: str, endpoint: str):
    """
    Crea un cliente boto3 con sesion fresca.
    La sesion se usa SOLO para generate_presigned_url (operacion local, sin red).
    """
    return b3s.Session().client(
        "s3",
        endpoint_url          = endpoint,
        aws_access_key_id     = access_key,
        aws_secret_access_key = secret_key,
        region_name           = "auto",
        config                = Config(
            signature_version = "s3v4",
            # Dict no-vacio (truthy): botocore lo usa directamente sin leer env vars.
            # Strings vacias = sin proxy para http y https.
            proxies = {"http": "", "https": ""},
        ),
    )


# ── Subida con requests (red directa, sin proxy) ──────────────────────────────

def _subir_con_requests(ruta_local: str, put_url: str) -> int:
    """
    Sube el archivo via HTTP PUT usando requests con proxies deshabilitados.

    proxies={"http": None, "https": None} con None (no cadena vacia):
      - None le dice a requests que ignore las vars de entorno para ese esquema.
      - Garantiza conexion directa aunque HTTP_PROXY/HTTPS_PROXY esten activas.

    Devuelve el HTTP status code (200 o 204 = exito en R2).
    """
    size_bytes = os.path.getsize(ruta_local)
    print(f"   📦 Tamano: {size_bytes / 1_048_576:.1f} MB")

    with open(ruta_local, "rb") as f:
        resp = requests.put(
            put_url,
            data    = f,
            headers = {"Content-Length": str(size_bytes)},
            proxies = {"http": None, "https": None},
            verify  = certifi.where(),
            timeout = 300,
        )

    print(f"   📡 HTTP {resp.status_code}")
    if resp.status_code not in (200, 204):
        body = resp.text[:600] if resp.text else "(sin cuerpo)"
        print(f"   ❌ Respuesta R2: {body}")

    return resp.status_code


# ── API publica ───────────────────────────────────────────────────────────────

def subir_clip_r2(ruta_local: str, nombre_archivo: str) -> "str | None":
    """
    Sube un clip a Cloudflare R2 y devuelve URL firmada valida 7 dias.
    Devuelve None si falla (no interrumpe el procesamiento de otros clips).

    Flujo:
      1. Diagnotsico de proxy ANTES de limpiar (para ver que tenia el entorno)
      2. Limpia vars de proxy + inyecta NO_PROXY=*
      3. Diagnostico DESPUES de limpiar (confirma que quedaron limpias)
      4. Crea cliente boto3 (sin red)
      5. Genera presigned PUT URL (local, sin red, sin SSL)
      6. Sube con requests + proxies=None (unica llamada de red)
      7. Genera presigned GET URL (local, sin red)
      8. Restaura vars de proxy para yt-dlp
    """
    access_key = os.environ.get("R2_ACCESS_KEY_ID",     "").strip()
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    endpoint   = os.environ.get("R2_ENDPOINT",          "").strip()
    bucket     = os.environ.get("R2_BUCKET", "klypo-clips").strip()

    if not all([access_key, secret_key, endpoint]):
        print("⚠️  R2: faltan R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ENDPOINT")
        return None

    ruta_real = _encontrar_archivo(ruta_local)
    if not ruta_real:
        return None

    clave_r2 = os.path.basename(ruta_real)
    print(f"\n📤 R2 subida: {clave_r2}")

    # Diagnostico ANTES de tocar el entorno
    _log_proxy_estado("ANTES-limpiar")
    guardadas = _limpiar_proxy()
    _log_proxy_estado("DESPUES-limpiar")

    try:
        s3 = _crear_cliente_s3(access_key, secret_key, endpoint)

        for intento in range(1, 4):
            try:
                print(f"   🔄 Intento {intento}/3")

                # Presigned PUT URL — operacion LOCAL, sin red, sin SSL
                put_url = s3.generate_presigned_url(
                    "put_object",
                    Params    = {"Bucket": bucket, "Key": clave_r2},
                    ExpiresIn = 3600,
                )

                # Subida real con requests, proxy=None explicito
                status = _subir_con_requests(ruta_real, put_url)
                if status not in (200, 204):
                    raise RuntimeError(f"R2 PUT devolvio HTTP {status}")

                # Presigned GET URL para devolver al cliente — operacion LOCAL
                get_url = s3.generate_presigned_url(
                    "get_object",
                    Params    = {"Bucket": bucket, "Key": clave_r2},
                    ExpiresIn = 604800,
                )
                print(f"   ✅ R2 OK: {get_url[:80]}...")
                return get_url

            except Exception as e:
                print(f"   ⚠️  Intento {intento} fallido: {type(e).__name__}: {e}")
                traceback.print_exc()
                if intento < 3:
                    time.sleep(2 * intento)

        print(f"   ❌ {clave_r2}: fallo definitivo tras 3 intentos")
        return None

    except Exception as e:
        print(f"❌ R2: error inesperado: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    finally:
        _restaurar_proxy(guardadas)
