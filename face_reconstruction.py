import cv2
import mediapipe as mp
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from plyfile import PlyData, PlyElement
from scipy.stats import zscore # NEW LIB
import re


def remove_coordinate_outliers(points3D):
    z_scores = np.abs(zscore(points3D, axis=0))

    # Replace outlier coordinates with NaN
    outlier_mask = z_scores >= 50
    points3D[outlier_mask] = np.nan

    return points3D


def scale_coordinates(points3D, scale_x=(-1, 1), scale_y=(-1.25, 1.25), scale_z=(-1.3, 1.3)):
    nan_mask = ~np.isnan(points3D)
    scaled_points = np.full(points3D.shape, np.nan)

    # Define the scale for each axis
    scales = [scale_x, scale_y, scale_z]

    for dim in range(points3D.shape[1]):
        valid_values = points3D[nan_mask[:, dim], dim]
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
        scale_min, scale_max = scales[dim]

        # Perform scaling only on non-NaN values
        scaled_points[nan_mask[:, dim], dim] = (valid_values - min_val) / (max_val - min_val) * (scale_max - scale_min) + scale_min

    return scaled_points


def select_images(folder_path):
    # Sort the image files
    image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))], key=lambda x: int(re.findall("\d+", x)[0]))
    total_images = len(image_files)

    # Select the 5 most centered images
    center_index = total_images // 2
    centered_images = [image_files[max(0, center_index - 2)], image_files[max(0, center_index - 1)], image_files[center_index],\
         image_files[min(total_images - 1, center_index + 1)], image_files[max(0, center_index + 2)]]

    # Select the 4 fifthile images
    quartile_step = total_images // 5
    quartile_images = [image_files[quartile_step], image_files[2*quartile_step], image_files[3*quartile_step], image_files[4*quartile_step]]

    # Combine all selected images and sort them
    selected_images = centered_images + quartile_images
    selected_images_sorted = sorted(selected_images, key=lambda x: int(re.findall("\d+", x)[0]))

    return selected_images_sorted

def get_keypoint_color(image, x, y):
    # Assuming x, y are the coordinates of the keypoint
    # Get the color at (x, y), making sure to convert from BGR to RGB
    color_bgr = image[int(y), int(x), :]  # OpenCV uses BGR by default
    color_rgb = color_bgr[::-1]  # Convert BGR to RGB
    return color_rgb


def interpolate_between_landmarks(landmarks, start_index, end_index, num_points=100):
    # Linearly interpolate points between two landmarks
    interpolated_points = []
    start_point = landmarks[start_index]
    end_point = landmarks[end_index]
    for i in range(1, num_points + 1):
        alpha = i / (num_points + 1)
        x = (1 - alpha) * start_point[0] + alpha * end_point[0]
        y = (1 - alpha) * start_point[1] + alpha * end_point[1]
        interpolated_points.append((x, y))
    return interpolated_points


def calculate_eye_region_3D(mean_points3D, start_index, end_index, num_points=100):
    # Get the 3D coordinates for the start and end points of the eye region
    start_point_3D = mean_points3D[start_index]
    end_point_3D = mean_points3D[end_index]
    
    # Interpolate points in 3D space between the start and end points
    interpolated_points_3D = []
    for i in range(1, num_points + 1):
        alpha = i / (num_points + 1)
        interpolated_point_3D = (1 - alpha) * start_point_3D + alpha * end_point_3D
        interpolated_points_3D.append(interpolated_point_3D)
    
    return np.array(interpolated_points_3D)


###################################################################################################################################################
# Esta clase 'PointCloudReconstructor' está diseñada para procesar un conjunto de imágenes y reconstruir una nube de puntos 3D utilizando OpenCV y MediaPipe.
# La clase utiliza la detección de puntos faciales de MediaPipe y descriptores SIFT de OpenCV para el matching y reconstruir la estructura 3D.
# Ofrece funcionalidades para visualizar la nube de puntos 3D, guardarla en formato PLY y visualizar un mallado 3D usando PyVista.
# Requiere un conjunto de imágenes almacenadas en un directorio específico y la matriz intrínseca de la cámara para realizar la reconstrucción 3D.

# Requisitos:
# - Directorio con imágenes.
# - Matriz intrínseca de la cámara.
# - Librerías Python: cv2 (OpenCV), mediapipe, numpy, matplotlib, pyvista, plyfile, scipy.
###################################################################################################################################################


class PointCloudReconstructor:
    def __init__(self, folder_path, original_image_dir, camera_matrix, max_slope_ratio=0.1):
        # Inicialización de la clase con la ruta del folder, la matriz de la cámara y el ratio máximo de pendiente.
        self.folder_path = folder_path
        self.original_image_dir = original_image_dir
        self.camera_matrix = camera_matrix
        self.max_slope_ratio = max_slope_ratio
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) #refine_landmarks=True

    def process_images(self):
        image_files = select_images(self.folder_path)
        # Assume the central image is the one with the best view
        central_image_path = os.path.join(self.original_image_dir, image_files[len(image_files) // 2])
        central_image = cv2.imread(central_image_path)
        points3D_list = []  # List to accumulate 3D points from each pair

        # Obtain landmarks and descriptors for the central image
        gray_central = cv2.cvtColor(central_image, cv2.COLOR_BGR2GRAY)
        landmarks_central = self.get_landmarks(gray_central)

        # After getting landmarks_central
        left_eye_interior_landmarks1 = interpolate_between_landmarks(landmarks_central, 161, 154)
        left_eye_interior_landmarks2 = interpolate_between_landmarks(landmarks_central, 158, 144)
        left_eye_interior_landmarks3 = interpolate_between_landmarks(landmarks_central, 157, 163)
        right_eye_interior_landmarks1 = interpolate_between_landmarks(landmarks_central, 398, 466)
        right_eye_interior_landmarks2 = interpolate_between_landmarks(landmarks_central, 385, 373)
        right_eye_interior_landmarks3 = interpolate_between_landmarks(landmarks_central, 387, 380)


        # Add the new landmarks for the eyes to the central landmarks
        landmarks_central += left_eye_interior_landmarks1 + right_eye_interior_landmarks1\
            + left_eye_interior_landmarks2 + right_eye_interior_landmarks2\
            + left_eye_interior_landmarks3 + right_eye_interior_landmarks3

        keypoints_central, descriptors_central = self.get_sift_descriptors(gray_central, landmarks_central)
        colors_central = [get_keypoint_color(central_image, kp.pt[0], kp.pt[1]) for kp in keypoints_central]

        

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
            
            # Matching
            good_matches = self.find_good_matches(keypoints1, keypoints2, descriptors1, descriptors2)
 
            #if len(good_matches) > 20:
            if len(keypoints1) != 0 and len(keypoints2) != 0:
                E = self.estimate_essential_matrix(keypoints1, keypoints2, self.camera_matrix)
                points3D = self.reconstruct_structure(E, keypoints1, keypoints2, self.camera_matrix)
                if points3D is not None:
                    points3D_with_nan = remove_coordinate_outliers(points3D)
                    scaled_points = scale_coordinates(points3D_with_nan)
                    points3D_list.append(scaled_points)
                    #average_colors = [(c1 + c2) // 2 for c1, c2 in zip(colors1, colors2)]  # Average colors
                    #colors_list.append(average_colors)
                    print(image1_path, image2_path)

        # Compute the mean of the 3D points if there are any reconstructions
        if points3D_list:
            # Assuming all points3D arrays are of the same shape
            mean_points3D = np.nanmean(np.array(points3D_list), axis=0)

            # Assuming you've already computed mean_points3D...
            left_eye_points_3D_1 = calculate_eye_region_3D(mean_points3D, 161, 173)
            left_eye_points_3D_2 = calculate_eye_region_3D(mean_points3D, 158, 144)
            left_eye_points_3D_3 = calculate_eye_region_3D(mean_points3D, 157, 163)
            right_eye_points_3D_1 = calculate_eye_region_3D(mean_points3D, 398, 466)
            right_eye_points_3D_2 = calculate_eye_region_3D(mean_points3D, 385, 373)
            right_eye_points_3D_3 = calculate_eye_region_3D(mean_points3D, 387, 380)

            # Concatenate the eye points to the mean_points3D for visualization
            visual_points3D = np.concatenate((mean_points3D, left_eye_points_3D_1, right_eye_points_3D_1,\
                                              left_eye_points_3D_2, right_eye_points_3D_2,\
                                                left_eye_points_3D_3, right_eye_points_3D_3,\
                                              ), axis=0)

            # Use colors from the central image directly
            self.display_point_cloud(visual_points3D)
            self.save_point_cloud_to_ply(visual_points3D, colors_central, 'nube_de_puntos.ply')
            self.display_mesh(visual_points3D, colors_central)
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

    def save_point_cloud_to_ply(self, points3D, colors, filename):
        # Make sure colors are in the correct format (uint8)
        colors = np.asarray(colors, dtype=np.uint8)
        vertex = np.array([(p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(points3D, colors)], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(filename)

    def display_mesh(self, points3D, colors):
        # Create a mesh from the point cloud
        cloud = pv.PolyData(points3D)
        colors = np.asarray(colors, dtype=np.uint8)
        cloud['colors'] = colors

        # Generate a surface mesh using Delaunay 2D
        surf = cloud.delaunay_2d()

        # Apply a mesh smoothing filter
        smooth_surf = surf.smooth(n_iter=75)  # Adjust n_iter for more or less smoothing n_iter=100

        # Plot the smoothed surface with colors
        plotter = pv.Plotter()
        plotter.add_mesh(smooth_surf, show_edges=False, rgb=True, scalars='colors')
        plotter.show()





