from descargador import descargar_video_youtube
from transcriptor import transcribir_segmento, transcribir_clip_timestamps, buscar_ganchos_en_segmento
from camara import precalcular_posiciones, nero_reframe
from subtitulos import agregar_subtitulos
from editor import ejecutar_nero

if __name__ == "__main__":
    ejecutar_nero()
