"""
KLYPO - Subida de clips a Cloudflare R2

Variables de entorno requeridas en RunPod:
  R2_ACCESS_KEY_ID     - Access Key ID del token de API de R2
  R2_SECRET_ACCESS_KEY - Secret Access Key del token de API de R2
  R2_ENDPOINT          - https://<ACCOUNT_ID>.r2.cloudflarestorage.com
  R2_BUCKET            - nombre del bucket (ej: klypo-clips)
"""

import os
import glob
import time
import boto3
from botocore.config import Config

# Variables de entorno de proxy que boto3 hereda y que rompen el SSL con R2.
# El proxy residencial es SOLO para yt-dlp, nunca para R2.
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
               "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")


def _crear_cliente_s3(access_key, secret_key, endpoint, bucket):
    """Crea el cliente boto3 con conexion directa a R2, sin proxy."""
    # 1. Elimina variables de proxy del entorno durante toda la operacion
    guardadas = {v: os.environ.pop(v) for v in _PROXY_VARS if v in os.environ}

    # 2. NO_PROXY=* dice a urllib3 que bypasee el proxy para CUALQUIER host.
    #    Esto cubre el caso donde RunPod inyecta el proxy a nivel de sistema
    #    y botocore lo lee despues de crear el cliente.
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"

    try:
        # 3. Sesion fresca — evita reutilizar cualquier estado cacheado de sesiones previas
        import boto3.session as b3s
        sesion = b3s.Session()
        cliente = sesion.client(
            "s3",
            endpoint_url          = endpoint,
            aws_access_key_id     = access_key,
            aws_secret_access_key = secret_key,
            region_name           = "auto",
            config                = Config(
                signature_version = "s3v4",
                # Dict no-vacio (truthy): botocore lo usa en vez de caer al fallback de env vars.
                # Strings vacias = sin proxy para http y https.
                proxies = {"http": "", "https": ""},
            ),
        )
        return cliente, guardadas
    except Exception:
        os.environ.pop("NO_PROXY", None)
        os.environ.pop("no_proxy", None)
        os.environ.update(guardadas)
        raise


def _restaurar_proxy(guardadas: dict):
    # Primero elimina los NO_PROXY que inyectamos (si no estaban antes no deben quedar)
    os.environ.pop("NO_PROXY", None)
    os.environ.pop("no_proxy", None)
    # Luego restaura todo lo que estaba originalmente (incluye NO_PROXY si existia)
    os.environ.update(guardadas)


def _encontrar_archivo(ruta_esperada: str) -> str | None:
    """
    Devuelve la ruta real del archivo aunque el nombre difiera levemente.
    Primero intenta la ruta exacta; si no existe, busca por prefijo en el mismo directorio.
    """
    if os.path.exists(ruta_esperada):
        return ruta_esperada

    directorio = os.path.dirname(ruta_esperada)
    nombre     = os.path.basename(ruta_esperada)

    # Busca por los primeros 30 caracteres del nombre (antes de que los titulos difieran)
    prefijo = nombre[:30]
    patron  = os.path.join(directorio, prefijo + "*.mp4")
    candidatos = glob.glob(patron)
    if candidatos:
        encontrado = candidatos[0]
        print(f"⚠️  Ruta exacta no hallada, usando: {os.path.basename(encontrado)}")
        return encontrado

    print(f"❌ Archivo no encontrado en disco: {ruta_esperada}")
    return None


def subir_clip_r2(ruta_local: str, nombre_archivo: str) -> str | None:
    """
    Sube un clip a Cloudflare R2 y devuelve una URL firmada valida por 7 dias.
    - Conexion directa sin proxy (el proxy es solo para yt-dlp).
    - Si la ruta exacta no existe, intenta encontrar el archivo por prefijo.
    - Reintenta hasta 3 veces si la subida falla.
    - Devuelve None si falla definitivamente (no interrumpe el proceso).
    """
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    endpoint   = os.environ.get("R2_ENDPOINT", "").strip()
    bucket     = os.environ.get("R2_BUCKET", "klypo-clips").strip()

    if not all([access_key, secret_key, endpoint, bucket]):
        print("⚠️  R2: faltan variables (R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET)")
        return None

    # Localiza el archivo real en disco
    ruta_real = _encontrar_archivo(ruta_local)
    if not ruta_real:
        return None

    # Usa el basename del archivo real como clave en R2
    clave_r2 = os.path.basename(ruta_real)

    # Crea el cliente sin proxy
    try:
        s3, guardadas_proxy = _crear_cliente_s3(access_key, secret_key, endpoint, bucket)
    except Exception as e:
        print(f"❌ R2: no se pudo crear cliente boto3: {e}")
        return None

    try:
        for intento in range(1, 4):
            try:
                print(f"⬆️  Subiendo a R2 (intento {intento}/3): {clave_r2}")
                s3.upload_file(
                    ruta_real,
                    bucket,
                    clave_r2,
                    ExtraArgs={"ContentType": "video/mp4"},
                )
                url = s3.generate_presigned_url(
                    "get_object",
                    Params    = {"Bucket": bucket, "Key": clave_r2},
                    ExpiresIn = 604800,   # 7 dias (maximo de R2)
                )
                print(f"✅ R2 OK: {url[:80]}...")
                return url
            except Exception as e:
                print(f"⚠️  R2 intento {intento} fallido: {e}")
                if intento < 3:
                    time.sleep(2 * intento)
        print(f"❌ R2: {clave_r2} no se pudo subir tras 3 intentos")
        return None
    finally:
        _restaurar_proxy(guardadas_proxy)
