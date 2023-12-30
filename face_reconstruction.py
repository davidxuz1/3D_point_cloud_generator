import cv2
import mediapipe as mp
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from plyfile import PlyData, PlyElement

###################################################################################################################################################
# Esta clase 'PointCloudReconstructor' está diseñada para procesar un conjunto de imágenes y reconstruir una nube de puntos 3D utilizando OpenCV y MediaPipe.
# La clase utiliza la detección de puntos faciales de MediaPipe y descriptores SIFT de OpenCV para el matching y reconstruir la estructura 3D.
# Ofrece funcionalidades para visualizar la nube de puntos 3D, guardarla en formato PLY y visualizar un mallado 3D usando PyVista.
# Requiere un conjunto de imágenes almacenadas en un directorio específico y la matriz intrínseca de la cámara para realizar la reconstrucción 3D.

# Requisitos:
# - Directorio con imágenes.
# - Matriz intrínseca de la cámara.
# - Librerías Python: cv2 (OpenCV), mediapipe, numpy, matplotlib, pyvista, plyfile.
###################################################################################################################################################

class PointCloudReconstructor:
    def __init__(self, folder_path, camera_matrix, max_slope_ratio=0.1):
        # Inicialización de la clase con la ruta del folder, la matriz de la cámara y el ratio máximo de pendiente.
        self.folder_path = folder_path
        self.camera_matrix = camera_matrix
        self.max_slope_ratio = max_slope_ratio
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    def process_images(self):
        # Procesa las imágenes en el directorio especificado y reconstruye la estructura 3D.
        image_files = sorted([f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))], key=lambda x: int(re.findall("\d+", x)[0]))
        points3D = None

        # Procesamiento de cada par de imágenes para la reconstrucción 3D.
        for i in range(len(image_files) - 1):
            image1_path = os.path.join(self.folder_path, image_files[i])
            image2_path = os.path.join(self.folder_path, image_files[i + 1])
            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

            # Obtención de landmarks y descriptores SIFT.
            landmarks1 = self.get_landmarks(gray1)
            landmarks2 = self.get_landmarks(gray2)
            keypoints1, descriptors1 = self.get_sift_descriptors(gray1, landmarks1)
            keypoints2, descriptors2 = self.get_sift_descriptors(gray2, landmarks2)
            good_matches = self.find_good_matches(keypoints1, keypoints2, descriptors1, descriptors2)

            if len(good_matches) > 20:
                # Estimación de la matriz esencial y reconstrucción de la estructura 3D.
                E = self.estimate_essential_matrix(keypoints1, keypoints2, self.camera_matrix)
                points3D = self.reconstruct_structure(E, keypoints1, keypoints2, self.camera_matrix)
                break

        # Visualización y guardado de la nube de puntos y visualización del mallado.
        if points3D is not None:
            self.display_point_cloud(points3D)
            self.save_point_cloud_to_ply(points3D, 'nube_de_puntos.ply')
            self.display_mesh(points3D)
        else:
            print("No se pudo reconstruir la estructura 3D.")

    def get_landmarks(self, image):
        # Procesa la imagen usando MediaPipe para detectar puntos faciales (landmarks).
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Si no se encuentran landmarks, retorna una lista vacía.
        if not results.multi_face_landmarks:
            return []
        # Convierte los landmarks detectados a coordenadas de píxeles y los retorna.
        return [(int(p.x * image.shape[1]), int(p.y * image.shape[0])) for p in results.multi_face_landmarks[0].landmark]

    def get_sift_descriptors(self, image, landmarks):
        # Crea un objeto SIFT para la extracción de descriptores.
        sift = cv2.SIFT_create()
        # Convierte los landmarks en keypoints para SIFT.
        keypoints = [cv2.KeyPoint(x, y, 1) for x, y in landmarks]
        # Calcula los descriptores SIFT para los keypoints y los retorna junto con los keypoints.
        keypoints, descriptors = sift.compute(image, keypoints)
        return keypoints, descriptors

    def find_good_matches(self, keypoints1, keypoints2, descriptors1, descriptors2):
        # Verifica si los descriptores están vacíos o si no son del mismo tipo.
        if descriptors1 is None or descriptors2 is None or descriptors1.dtype != descriptors2.dtype:
            return []
        # Crea un objeto BFMatcher para encontrar correspondencias entre descriptores.
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        # Filtra los matches basándose en la distancia y la pendiente máxima permitida.
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                pt1 = keypoints1[m.queryIdx].pt
                pt2 = keypoints2[m.trainIdx].pt
                slope = float('inf') if pt2[0] - pt1[0] == 0 else abs((pt2[1] - pt1[1]) / (pt2[0] - pt1[0]))
                if slope < self.max_slope_ratio:
                    good_matches.append(m)
        return good_matches

    def estimate_essential_matrix(self, keypoints1, keypoints2, K):
        # Convierte los keypoints a formato NumPy float32.
        points1 = np.float32([kp.pt for kp in keypoints1])
        points2 = np.float32([kp.pt for kp in keypoints2])
        # Calcula la matriz esencial entre los pares de puntos y la retorna.
        return cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)[0]

    def reconstruct_structure(self, E, keypoints1, keypoints2, K):
        # Recupera la pose relativa entre dos vistas a partir de la matriz esencial.
        _, R, t, _ = cv2.recoverPose(E, np.float32([kp.pt for kp in keypoints1]), np.float32([kp.pt for kp in keypoints2]), K)
        # Prepara las matrices de proyección para las dos vistas.
        P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))
        # Triangula los puntos en el espacio 3D y los normaliza.
        points4D_hom = cv2.triangulatePoints(K @ P1, K @ P2, np.float32([kp.pt for kp in keypoints1]).T, np.float32([kp.pt for kp in keypoints2]).T)
        points4D = points4D_hom / points4D_hom[3]
        return points4D[:3].T

    def display_point_cloud(self, points3D):
        # Crea una figura 3D para visualizar la nube de puntos.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2])
        ax.set_xlabel('Eje X')
        ax.set_ylabel('Eje Y')
        ax.set_zlabel('Eje Z')
        plt.show()

    def save_point_cloud_to_ply(self, points3D, filename):
        # Prepara y guarda la nube de puntos en un archivo PLY.
        vertex = np.array([(p[0], p[1], p[2]) for p in points3D], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(filename)

    def display_mesh(self, points3D):
        # Crea una malla a partir de la nube de puntos y la visualiza.
        cloud = pv.PolyData(points3D)
        surf = cloud.delaunay_2d()
        surf.plot(show_edges=True)


