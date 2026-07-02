"""
KLYPO - Subida de clips a Cloudflare R2

Variables requeridas en RunPod:
  R2_ACCESS_KEY_ID     - Access Key ID del token API de R2
  R2_SECRET_ACCESS_KEY - Secret Access Key del token API de R2
  R2_ENDPOINT          - https://<ACCOUNT_ID>.r2.cloudflarestorage.com
  R2_BUCKET            - nombre del bucket (ej: klypo-clips)

Arquitectura de subida:
  1. boto3 genera la presigned PUT URL LOCALMENTE (firma criptografica, sin red).
  2. requests.put() sube el archivo con proxies=None y un HTTPAdapter que inyecta
     un ssl.SSLContext explicito.  Esto bypasea tanto el proxy como el openssl.cnf
     del sistema (que en Ubuntu 22.04 fuerza SECLEVEL=2 y restringe los ciphers).
  3. Si el primer ssl_context falla con SSLError, se prueba el siguiente en cascade
     hasta encontrar uno compatible con el edge de Cloudflare que atiende la request.
"""

import os
import glob
import ssl
import time
import traceback

import boto3.session as b3s
import certifi
import requests
import urllib3
from requests.adapters import HTTPAdapter
from botocore.config import Config


# Variables de entorno que pueden desviar trafico por un proxy
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
               "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")


# ── HTTPAdapter con ssl_context inyectable ────────────────────────────────────

class _TLSAdapter(HTTPAdapter):
    """
    HTTPAdapter que monta un ssl.SSLContext personalizado en el PoolManager.
    Permite controlar version TLS, cipher suites y CA bundle de forma independiente
    del openssl.cnf del sistema operativo.
    """

    def __init__(self, ssl_context: ssl.SSLContext, **kwargs):
        self._ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, num_pools, maxsize, block=False, **kw):
        # ssl_context se pasa a traves de kw hasta urllib3.HTTPSConnectionPool
        kw["ssl_context"] = self._ssl_context
        super().init_poolmanager(num_pools, maxsize, block=block, **kw)


# ── Fabrica de ssl contexts ───────────────────────────────────────────────────

def _ctx(min_ver=ssl.TLSVersion.TLSv1_2,
         max_ver: "ssl.TLSVersion | None" = None,
         ciphers: "str | None" = None) -> ssl.SSLContext:
    """Crea ssl.SSLContext con los parametros dados. Siempre usa certifi como CA."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.minimum_version = min_ver
    if max_ver is not None:
        ctx.maximum_version = max_ver
    if ciphers is not None:
        ctx.set_ciphers(ciphers)
    return ctx


def _ctx_urllib3() -> ssl.SSLContext:
    """
    Contexto creado por urllib3 — diferentes defaults que ssl stdlib.
    A veces resuelve incompatibilidades cuando el contexto Python falla.
    """
    from urllib3.util.ssl_ import create_urllib3_context
    ctx = create_urllib3_context()
    ctx.load_verify_locations(cafile=certifi.where())
    return ctx


# Cascade de configuraciones TLS.
# Se prueban en orden; en cuanto una supera el handshake se usa para el resto del intento.
# Formato: (etiqueta_para_logs, callable_que_devuelve_ssl_context)
_TLS_CASCADE = [
    # 1. TLS 1.2+ con ciphers por defecto del sistema — funciona en la mayoria de casos
    ("TLS-1.2+  ciphers-default",
     lambda: _ctx()),

    # 2. Forzar TLS 1.2 exacto — descarta TLS 1.3 por si el edge europeo lo rechaza
    ("TLS-1.2   forzado",
     lambda: _ctx(max_ver=ssl.TLSVersion.TLSv1_2)),

    # 3. SECLEVEL=1 — openssl.cnf de Ubuntu 22.04 usa SECLEVEL=2 que puede restringir
    #    los cipher suites ofrecidos en el ClientHello hasta que el servidor rechaza
    ("TLS-1.2+  SECLEVEL=1",
     lambda: _ctx(ciphers="DEFAULT@SECLEVEL=1")),

    # 4. Contexto de urllib3 — distinto al de ssl stdlib, a veces resuelve edge cases
    ("TLS-1.2+  urllib3-ctx",
     _ctx_urllib3),
]


# ── Proxy: diagnostico y gestion ─────────────────────────────────────────────

def _log_proxy(tag: str):
    """Imprime el estado real de cada variable de proxy en os.environ."""
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


# ── Subida HTTP con cascade TLS ───────────────────────────────────────────────

def _subir_con_tls_cascade(ruta_local: str, put_url: str) -> int:
    """
    Sube el archivo intentando cada config TLS en orden.

    - SSLError -> continua al siguiente config de la cascade
    - HTTP error (no SSL) -> lanza inmediatamente (no es problema de TLS)
    - Si todos los configs fallan con SSLError -> lanza RuntimeError con el resumen

    Devuelve el HTTP status code cuando tiene exito (200 o 204 para R2).
    """
    size = os.path.getsize(ruta_local)
    print(f"   📦 Tamano: {size / 1_048_576:.1f} MB")

    ssl_errores = []

    for etiqueta, make_ctx in _TLS_CASCADE:
        try:
            print(f"   🔒 TLS: {etiqueta}")
            ctx     = make_ctx()
            session = requests.Session()
            session.mount("https://", _TLSAdapter(ctx))

            with open(ruta_local, "rb") as f:
                resp = session.put(
                    put_url,
                    data    = f,
                    headers = {"Content-Length": str(size)},
                    proxies = {"http": None, "https": None},
                    timeout = 300,
                )

            print(f"   📡 HTTP {resp.status_code}")
            if resp.status_code in (200, 204):
                return resp.status_code

            # Error HTTP (no TLS) — no tiene sentido probar otro ssl_context
            body = (resp.text or "")[:500]
            print(f"   ❌ R2 body: {body}")
            raise RuntimeError(f"R2 devolvio HTTP {resp.status_code}: {body[:200]}")

        except (ssl.SSLError, requests.exceptions.SSLError) as e:
            msg = str(e)
            print(f"   ❌ SSLError ({etiqueta}): {msg}")
            ssl_errores.append(f"[{etiqueta}] {msg}")
            # Continuar al siguiente ssl_context

        except RuntimeError:
            raise  # HTTP error — propagar directamente

        except Exception as e:
            # Error de red no-SSL — propagar con contexto
            print(f"   ❌ Error inesperado ({etiqueta}): {type(e).__name__}: {e}")
            raise

    # Todos los ssl_contexts fallaron
    resumen = " | ".join(ssl_errores)
    raise RuntimeError(f"SSLV3_ALERT en todas las configs TLS: {resumen}")


# ── API publica ───────────────────────────────────────────────────────────────

def subir_clip_r2(ruta_local: str, nombre_archivo: str) -> "str | None":
    """
    Sube un clip a Cloudflare R2. Devuelve URL firmada valida 7 dias o None.

    Flujo:
      1. Diagnostico de proxy (ANTES y DESPUES de limpiar)
      2. Genera presigned PUT URL con boto3 — operacion LOCAL, sin red, sin TLS
      3. Sube con requests + cascade TLS + proxies=None
      4. Genera presigned GET URL con boto3 — operacion LOCAL, sin red, sin TLS
      5. Restaura vars de proxy para yt-dlp
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

    try:
        # Sesion fresca — evita estado cacheado de sesiones boto3 anteriores
        s3 = b3s.Session().client(
            "s3",
            endpoint_url          = endpoint,
            aws_access_key_id     = access_key,
            aws_secret_access_key = secret_key,
            region_name           = "auto",
            config                = Config(
                signature_version = "s3v4",
                proxies           = {"http": "", "https": ""},  # truthy: no lee env vars
            ),
        )

        for intento in range(1, 4):
            try:
                print(f"   🔄 Intento {intento}/3")

                # Presigned PUT — firma criptografica LOCAL, sin ninguna llamada de red
                put_url = s3.generate_presigned_url(
                    "put_object",
                    Params    = {"Bucket": bucket, "Key": clave_r2},
                    ExpiresIn = 3600,
                )

                # Unica llamada de red: requests con cascade TLS y proxy=None
                _subir_con_tls_cascade(ruta_real, put_url)

                # Presigned GET — firma LOCAL, sin red
                get_url = s3.generate_presigned_url(
                    "get_object",
                    Params    = {"Bucket": bucket, "Key": clave_r2},
                    ExpiresIn = 604800,  # 7 dias (maximo de R2)
                )
                print(f"   ✅ R2 OK: {get_url[:80]}...")
                return get_url

            except Exception as e:
                print(f"   ⚠️  Intento {intento} error: {type(e).__name__}: {e}")
                traceback.print_exc()
                if intento < 3:
                    time.sleep(2 * intento)

        print(f"   ❌ {clave_r2}: fallo definitivo tras 3 intentos")
        return None

    except Exception as e:
        print(f"❌ R2 error inesperado: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    finally:
        _restaurar_proxy(guardadas)
