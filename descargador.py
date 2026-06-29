import os
import yt_dlp


def descargar_video_youtube(url):
    base_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'video_descargado.mp4',
        'overwrites': True,
        'retries': 10,
        'fragment_retries': 10,
        'continuedl': True,
        'remote_components': 'ejs:github',
    }

    # Intento 1: llave.txt si pesa más de 3KB
    llave_ok = os.path.exists('llave.txt') and os.path.getsize('llave.txt') > 3000
    if llave_ok:
        print(f"📥 KLYPO usando llave.txt para: {url}")
        try:
            opts = {**base_opts, 'cookiefile': 'llave.txt'}
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            return "video_descargado.mp4"
        except Exception as e:
            print(f"⚠️ llave.txt falló ({e}) — intentando con cookies de Chrome...")
    else:
        print(f"⚠️ llave.txt no encontrado o muy pequeño ({os.path.getsize('llave.txt') if os.path.exists('llave.txt') else 0} bytes)")

    # Intento 2: cookies directo desde Chrome/Firefox/Edge
    for browser in ('chrome', 'firefox', 'edge'):
        print(f"📥 Intentando con cookies de {browser}...")
        try:
            opts = {**base_opts, 'cookiesfrombrowser': (browser,)}
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            print(f"✅ Descargado con cookies de {browser}")
            return "video_descargado.mp4"
        except Exception as e:
            print(f"⚠️ {browser} falló: {e}")

    print("❌ No se pudo descargar con ningún método.")
    print("💡 Solución: exporta cookies frescas desde chrome con 'Get cookies.txt LOCALLY' y guárdalas como llave.txt (debe pesar >3KB)")
    return None
