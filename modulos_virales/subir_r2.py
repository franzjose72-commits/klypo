"""
KLYPO - Subida de clips a Cloudflare R2

Variables de entorno requeridas en RunPod:
  R2_ACCESS_KEY_ID     - Access Key ID del token de API de R2
  R2_SECRET_ACCESS_KEY - Secret Access Key del token de API de R2
  R2_ENDPOINT          - https://<ACCOUNT_ID>.r2.cloudflarestorage.com
  R2_BUCKET            - nombre del bucket (ej: klypo-clips)
"""

import os
import boto3
from botocore.client import Config


def subir_clip_r2(ruta_local: str, nombre_archivo: str) -> str | None:
    """
    Sube un clip a Cloudflare R2 y devuelve una URL firmada valida por 7 dias.
    Devuelve None si falla (el proceso no se interrumpe).
    """
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_key  = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    endpoint    = os.environ.get("R2_ENDPOINT", "").strip()
    bucket      = os.environ.get("R2_BUCKET", "klypo-clips").strip()

    if not all([access_key, secret_key, endpoint, bucket]):
        print("⚠️  R2: faltan variables de entorno (R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET)")
        return None

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url         = endpoint,
            aws_access_key_id    = access_key,
            aws_secret_access_key = secret_key,
            config               = Config(signature_version="s3v4"),
            region_name          = "auto",
        )

        print(f"⬆️  Subiendo a R2: {nombre_archivo}")
        s3.upload_file(
            ruta_local,
            bucket,
            nombre_archivo,
            ExtraArgs={"ContentType": "video/mp4"},
        )

        # URL firmada valida 7 dias (604800 segundos — maximo de R2)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": nombre_archivo},
            ExpiresIn=604800,
        )
        print(f"✅ R2 OK: {url[:80]}...")
        return url

    except Exception as e:
        print(f"❌ R2 error subiendo {nombre_archivo}: {e}")
        return None
