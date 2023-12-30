import cv2
import os

###################################################################################################################################################
# Este script define una clase FrameExtractor para extraer frames individuales de archivos de video.
# La clase tiene dos métodos principales: 'extract_frames' para extraer frames de un video específico y 
# 'process_videos' para procesar todos los videos en un directorio dado.
# 'extract_frames' toma un video, lo procesa frame por frame y guarda los frames seleccionados como imágenes en un directorio específico.
# 'process_videos' busca todos los archivos de video en el directorio 'raw_video' y utiliza 'extract_frames' para extraer y almacenar los frames.
# Los frames extraídos se guardan en un directorio estructurado según el nombre del video original.

# Requisitos:
# - Archivos de video almacenados en un directorio específico.
# - Librería Python: cv2 (OpenCV), os.
# pip install opencv-python
###################################################################################################################################################

class FrameExtractor:
    def __init__(self, raw_video_dir='./raw_video', extracted_frames_dir='./extracted_frames'):
        # Constructor de la clase: inicializa los directorios para videos crudos y frames extraídos
        self.raw_video_dir = raw_video_dir
        self.extracted_frames_dir = extracted_frames_dir
        # Crea el directorio para los frames extraídos si no existe
        os.makedirs(self.extracted_frames_dir, exist_ok=True)
    
    def extract_frames(self, video_path, output_dir, frame_rate=3):
        """
        Extrae frames de un archivo de video y los guarda como imágenes.
        """
        # Crea el directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        # Abre el video para procesamiento
        cap = cv2.VideoCapture(video_path)
        # Obtiene el número total de frames en el video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        extracted_frames = []

        # Recorre todos los frames del video
        for i in range(frame_count):
            ret, frame = cap.read()
            # Si no hay frame para leer, termina el bucle
            if not ret:
                break
            # Guarda cada frame según la tasa de frames especificada
            if i % frame_rate == 0:
                frame_path = os.path.join(output_dir, f"frame_{i}.png")
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)

        # Libera el objeto de captura de video
        cap.release()
        # Retorna una lista de los paths de los frames extraídos
        return extracted_frames
    
    def process_videos(self):
        # Procesa todos los archivos de video en el directorio de videos crudos
        for video_file in os.listdir(self.raw_video_dir):
            video_path = os.path.join(self.raw_video_dir, video_file)
            video_name_without_extension = os.path.splitext(video_file)[0]
            output_dir_for_video = os.path.join(self.extracted_frames_dir, video_name_without_extension)
            # Llama a extract_frames para cada video
            self.extract_frames(video_path, output_dir_for_video)

"""
# Crear una instancia de FrameExtractor y procesar los videos
if __name__ == "__main__":
    frame_extractor = FrameExtractor()
    frame_extractor.process_videos()
"""
