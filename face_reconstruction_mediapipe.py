import cv2
import mediapipe as mp
import numpy as np
import os
import pyvista as pv
from plyfile import PlyData, PlyElement

###################################################################################################################################################
# Este script utiliza OpenCV y MediaPipe para detectar puntos de referencia faciales en 3D a partir de una sola imagen.
# Se emplea MediaPipe Face Mesh para obtener 468 puntos de referencia faciales.
# Los puntos se utilizan para crear una nube de puntos y un archivo PLY.
# Además, se utiliza PyVista para realizar el mallado de la nube de puntos.

# Requisitos:
# - Una imagen específica para el procesamiento.
# - Librerías Python: cv2 (OpenCV), mediapipe, numpy, pyvista.
# pip install opencv-python mediapipe numpy pyvista opencv-contrib-python

# Input: Una imagen específica.
# Output: Archivo PLY y visualización del mallado de puntos faciales.
###################################################################################################################################################

# Función para obtener los 468 puntos de referencia de la cara
def get_468_landmarks(image):
    # Transformar imagen de BGR a RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Tamaño de la imagen
    height, width, _ = rgb_image.shape

    # Inicializar MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # Detectar puntos de referencia faciales
    result = face_mesh.process(rgb_image)

    landmarks = []
    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                point = facial_landmarks.landmark[i]
                x = point.x * width
                y = point.y * height
                z = point.z * width  # Profundidad en escala con el ancho para mantener la proporción
                landmarks.append((x, y, z))

    return landmarks

# Función para crear un archivo PLY a partir de puntos
def create_ply(points, filename):
    vertex = np.array([(p[0], p[1], p[2]) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(filename)

# Función para realizar el mallado usando PyVista
def mesh_points(points):
    cloud = pv.PolyData(points)
    surf = cloud.delaunay_2d()
    surf.plot(show_edges=True)

# Ruta a la imagen específica
image_path = 'extracted_frames/test/processed/frame_75.png'  # Reemplaza con la ruta a tu imagen
image = cv2.imread(image_path)

# Verifica si el archivo existe antes de intentar cargarlo
if os.path.exists(image_path):
    image = cv2.imread(image_path)

    # Verifica si la imagen se ha cargado correctamente
    if image is not None:
        landmarks = get_468_landmarks(image)
        # El resto de tu procesamiento aquí
    else:
        print(f"Error: No se pudo cargar la imagen desde {image_path}")
else:
    print(f"Error: El archivo no existe en la ruta {image_path}")

# Procesar la imagen, crear archivo PLY y realizar el mallado
landmarks = get_468_landmarks(image)
if landmarks:
    ply_filename = 'output.ply'
    create_ply(landmarks, ply_filename)
    mesh_points(landmarks)

