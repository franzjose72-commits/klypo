import os
from google import genai

# --- 1. CONFIGURACIÓN DIRECTA ---
# PEGA TU LLAVE NUEVA AQUÍ ABAJO (La que copiaste de Google AI Studio)
llave_directa = "AIzaSyDPw99HaCNxBLGwErqL7B58t35ECiHJLDc"


def encender_motor():
    print("--- 🚀 Lanzando el Motor de Franz (Modo Directo) ---")

    try:
        # 2. Configuramos el cliente con la llave pegada arriba
        client = genai.Client(api_key=llave_directa)

        # 3. Llamada al modelo 2.0 Flash
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Responde solo: DIME HOLA"
        )

        print(f"\n✅ ¡LO LOGRASTE, HERMANO! {response.text}")

    except Exception as e:
        print(f"\n❌ NOTA DEL SISTEMA: {e}")


if __name__ == "__main__":
    encender_motor()