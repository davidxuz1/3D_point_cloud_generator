import numpy as np
import os
from face_reconstruction import PointCloudReconstructor
from preprocessing import Preprocessor

###################################################################################################################################################
# Script para la reconstrucción de nubes de puntos 3D a partir de un vídeo de un rostro.
###################################################################################################################################################


# # Inicializa y procesa el video para obtener los frames.
# preprocessor = Preprocessor(video_dir='./raw_video', video_name='test_david.mp4')  # Nombre del video a procesar.
# processed_dir, original_image_dir = preprocessor.process_video()
# print(f"Las imágenes procesadas se guardan en: {processed_dir}")

# Si las imágenes procesadas ya están creadas, comentar lo de arriba y decomentar las dos líneas de abajo. Cambiando "test_vladys" por ""test_dani" u otro se puede ver la cara de otros.
processed_dir = "./extracted_frames/test_vladys/processed"
original_image_dir = "./extracted_frames/test_vladys"
# --- Parámetros de la Cámara (iPhone 13 mini) ---
"""
focal_length_mm = 26  # Longitud focal en mm 26
sensor_width_mm = 9.5  # Tamaño del sensor en mm (aproximado para 1/1.9")
image_resolution = (4000, 3000)  # Resolución de la imagen en píxeles (4000, 3000)
focal_length_px = focal_length_mm * image_resolution[0] / sensor_width_mm
cx, cy = image_resolution[0] / 2, image_resolution[1] / 2
K = np.array([[focal_length_px, 0, cx], [0, focal_length_px, cy], [0, 0, 1]]) # Matriz intrínseca K
"""
# Matriz intrínseca del sensor de vídeo del iPhone 13 mini con resolución (480, 848)
K = np.array([[1.97491666e+03, 0.00000000e+00, 3.15881152e+02],
              [0.00000000e+00, 1.92595086e+03, 5.12186986e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Crea una instancia de PointCloudReconstructor y procesa las imágenes. Retornar nube de puntos, guardar en un archivo .ply y mallado del rostro.
reconstructor = PointCloudReconstructor(processed_dir, original_image_dir, K)
reconstructor.process_images()