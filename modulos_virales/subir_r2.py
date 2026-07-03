"""
KLYPO - Subida de clips a Cloudflare R2 via curl

Variables requeridas en RunPod:
  R2_ACCESS_KEY_ID     - Access Key ID del token API de R2
  R2_SECRET_ACCESS_KEY - Secret Access Key del token API de R2
  R2_ENDPOINT          - https://<ACCOUNT_ID>.r2.cloudflarestorage.com
  R2_BUCKET            - nombre del bucket (ej: klypo-clips)

Estrategia:
  1. boto3 genera la presigned PUT URL LOCALMENTE (sin red, sin TLS — ya funciona).
  2. curl hace el PUT real. curl usa libcurl con sus propios defaults TLS,
     completamente independiente del ssl/OpenSSL de Python y del SECLEVEL=2
     que Ubuntu 22.04 inyecta via /etc/ssl/openssl.cnf.
  3. boto3 genera la presigned GET URL LOCALMENTE (sin red — ya funciona).
"""

import os
import glob
import subprocess
import time
import traceback

import boto3.session as b3s
import certifi
from botocore.config import Config


_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
               "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")


# ── Proxy: diagnostico y gestion ─────────────────────────────────────────────

def _log_proxy(tag: str):
    activas = {v: os.environ[v] for v in _PROXY_VARS if v in os.environ}
    if activas:
        print(f"   ⚠️  [{tag}] Proxy vars ACTIVAS:")
        for k, v in activas.items():
            print(f"         {k} = {v!r}")
    else:
        print(f"   ✅ [{tag}] Sin proxy en entorno")


def _limpiar_proxy() -> dict:
    guardadas = {v: os.environ.pop(v) for v in _PROXY_VARS if v in os.environ}
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    return guardadas


def _restaurar_proxy(guardadas: dict):
    os.environ.pop("NO_PROXY", None)
    os.environ.pop("no_proxy", None)
    os.environ.update(guardadas)


# ── Utilidades de archivo ─────────────────────────────────────────────────────

def _encontrar_archivo(ruta_esperada: str) -> "str | None":
    if os.path.exists(ruta_esperada):
        return ruta_esperada
    directorio = os.path.dirname(ruta_esperada)
    patron     = os.path.join(directorio, os.path.basename(ruta_esperada)[:30] + "*.mp4")
    candidatos = glob.glob(patron)
    if candidatos:
        print(f"   ⚠️  Ruta exacta no hallada, usando: {os.path.basename(candidatos[0])}")
        return candidatos[0]
    print(f"   ❌ Archivo no encontrado: {ruta_esperada}")
    return None


# ── Test TLS con curl (diagnostico, solo se ejecuta una vez por worker) ───────

_tls_test_hecho = False

def _test_tls_curl(endpoint: str):
    """
    Prueba si curl puede hacer el handshake TLS con el endpoint de R2.
    Imprime las lineas relevantes del verbose de curl para diagnosticar.
    Solo se ejecuta la primera vez (no por cada clip).
    """
    global _tls_test_hecho
    if _tls_test_hecho:
        return
    _tls_test_hecho = True

    print("   🔍 Test TLS curl vs R2 endpoint:")
    try:
        result = subprocess.run(
            [
                "curl", "-v",
                "--head",               # solo HEAD, no descarga cuerpo
                "--silent",             # sin barra de progreso
                "--max-time", "15",
                endpoint,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        # curl -v escribe el handshake TLS en stderr
        salida = result.stderr + "\n" + result.stdout
        palabras_clave = ("SSL", "TLS", "cipher", "Cipher", "Connected",
                          "CONNECTED", "< HTTP", "> HEAD", "handshake",
                          "certificate", "issuer", "subject", "alert", "ALERT")
        for linea in salida.splitlines():
            if any(k in linea for k in palabras_clave):
                print(f"      {linea}")
        if result.returncode == 0:
            print("   ✅ Test TLS curl: EXITO (handshake completado)")
        else:
            print(f"   ❌ Test TLS curl: FALLO (exit {result.returncode})")
    except Exception as e:
        print(f"   ⚠️  Test TLS curl: excepcion {type(e).__name__}: {e}")


# ── Subida via curl ───────────────────────────────────────────────────────────

def _subir_con_curl(ruta_local: str, put_url: str) -> bool:
    """
    Hace el HTTP PUT del archivo usando el binario curl del sistema.

    Por que curl en vez de Python requests/boto3:
      - curl usa libcurl con TLS propio.
      - No hereda SECLEVEL=2 de /etc/ssl/openssl.cnf (Ubuntu 22.04).
      - No usa el modulo ssl de Python ni su contexto por defecto.
      - Tiene su propio conjunto de cipher suites y TLS version negotiation.

    Devuelve True si el upload fue exitoso (HTTP 200 o 204), False si fallo.
    """
    size_mb = os.path.getsize(ruta_local) / 1_048_576
    print(f"   📦 Tamano: {size_mb:.1f} MB")

    cmd = [
        "curl",
        "-X", "PUT",
        "--data-binary", f"@{ruta_local}",  # @ = leer desde archivo (streaming)
        "-H", "Content-Type: video/mp4",
        "-w", "\nHTTP_CODE:%{http_code}",   # escribe el status al final de stdout
        "-o", "/dev/null",                  # descarta el cuerpo de la respuesta
        "--silent",                          # sin barra de progreso
        "--show-error",                      # pero si muestra errores de red
        "--max-time", "300",                 # 5 min timeout total
        "--connect-timeout", "30",           # 30 s para establecer conexion
        put_url,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=310,
        )

        # Extraer HTTP status code de stdout ("...HTTP_CODE:200")
        http_code = ""
        for linea in result.stdout.splitlines():
            if linea.startswith("HTTP_CODE:"):
                http_code = linea.split(":", 1)[1].strip()

        print(f"   📡 curl exit={result.returncode} HTTP={http_code or '(no code)'}")

        if result.stderr.strip():
            # Mostrar stderr de curl (errores de red, TLS, etc.)
            print(f"   📋 curl stderr: {result.stderr.strip()[:400]}")

        if result.returncode == 0 and http_code in ("200", "204"):
            return True

        print("   ❌ curl: upload no exitoso")
        return False

    except subprocess.TimeoutExpired:
        print("   ❌ curl timeout (>5 min)")
        return False
    except FileNotFoundError:
        print("   ❌ curl no encontrado en el contenedor")
        return False
    except Exception as e:
        print(f"   ❌ curl excepcion: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ── API publica ───────────────────────────────────────────────────────────────

def subir_clip_r2(ruta_local: str, nombre_archivo: str) -> "str | None":
    """
    Sube un clip a Cloudflare R2 usando curl para el PUT.
    Devuelve URL firmada valida 7 dias, o None si falla definitivamente.

    Flujo:
      1. Diagnostico de proxy (ANTES y DESPUES de limpiar)
      2. Test TLS curl vs endpoint (solo la primera vez — diagnostico)
      3. boto3 genera presigned PUT URL — LOCAL, sin red
      4. curl hace el PUT — red directa, TLS propio de libcurl
      5. boto3 genera presigned GET URL — LOCAL, sin red
      6. Restaura proxy para yt-dlp
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
    print(f"\n📤 R2: {clave_r2}")

    _log_proxy("ANTES-limpiar")
    guardadas = _limpiar_proxy()
    _log_proxy("DESPUES-limpiar")

    # Test TLS una sola vez por worker (informativo, no bloquea)
    _test_tls_curl(endpoint)

    try:
        # Sesion fresca — solo para firmar URLs localmente (sin red)
        # addressing_style="path" fuerza URLs como:
        #   https://ACCOUNT.r2.cloudflarestorage.com/BUCKET/KEY?X-Amz-...
        # Sin esto, boto3 puede usar virtual-hosted (BUCKET.ACCOUNT.r2...) que R2
        # no sirve correctamente para acceso publico via presigned GET URL.
        s3 = b3s.Session().client(
            "s3",
            endpoint_url          = endpoint,
            aws_access_key_id     = access_key,
            aws_secret_access_key = secret_key,
            region_name           = "auto",
            config                = Config(
                signature_version = "s3v4",
                proxies           = {"http": "", "https": ""},
                s3                = {"addressing_style": "path"},
            ),
        )

        for intento in range(1, 4):
            try:
                print(f"   🔄 Intento {intento}/3")

                # Presigned PUT: operacion LOCAL, sin red, sin TLS
                put_url = s3.generate_presigned_url(
                    "put_object",
                    Params    = {"Bucket": bucket, "Key": clave_r2},
                    ExpiresIn = 3600,
                )

                # PUT real via curl
                ok = _subir_con_curl(ruta_real, put_url)
                if not ok:
                    raise RuntimeError("curl upload fallido")

                # Presigned GET: operacion LOCAL, sin red, sin TLS
                # Si R2_PUBLIC_URL esta configurado (ej: https://pub-HASH.r2.dev),
                # devuelve URL publica directa sin firma — mas fiable para navegadores.
                r2_public = os.environ.get("R2_PUBLIC_URL", "").strip().rstrip("/")
                if r2_public:
                    get_url = f"{r2_public}/{clave_r2}"
                    print(f"   ✅ R2 OK (URL publica): {get_url}")
                else:
                    get_url = s3.generate_presigned_url(
                        "get_object",
                        Params    = {"Bucket": bucket, "Key": clave_r2},
                        ExpiresIn = 604800,  # 7 dias
                    )
                    # URL completa en log para poder inspeccionarla si falla en navegador
                    print(f"   ✅ R2 OK (presigned URL):")
                    print(f"      {get_url}")
                return get_url

            except Exception as e:
                print(f"   ⚠️  Intento {intento} error: {type(e).__name__}: {e}")
                if intento < 3:
                    time.sleep(2 * intento)

        print(f"   ❌ {clave_r2}: fallo tras 3 intentos")
        return None

    except Exception as e:
        print(f"❌ R2 error inesperado: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    finally:
        _restaurar_proxy(guardadas)
