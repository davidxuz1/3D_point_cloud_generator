import cv2
import mediapipe as mp
import numpy as np
import os
import re


###################################################################################################################################################
# Este script utiliza OpenCV y MediaPipe para detectar puntos faciales y realizar matching de imágenes.
# pip install opencv-python
# pip install opencv-contrib-python
# pip install mediapipe

# La carpeta './extracted_frames/test/processed' debe contener imágenes preprocesadas en formato .png para su procesamiento.
# Este script utiliza el modelo de malla facial de MediaPipe en lugar del shape_predictor_68_face_landmarks.dat de dlib.

# Input: Frames de imágenes preprocesadas.
# Output: Matching de dos frames elegidos, utilizando descriptores SIFT y keypoints detectados con MediaPipe.
###################################################################################################################################################

# MediaPipe face mesh detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Function to detect facial landmarks using MediaPipe
def get_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return []
    landmarks = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in results.multi_face_landmarks[0].landmark]
    return landmarks

# Function to get SIFT descriptors from keypoints obtained from the model
def get_sift_descriptors(image, landmarks):
    sift = cv2.SIFT_create()
    keypoints = [cv2.KeyPoint(x, y, 1) for (x, y) in landmarks]
    keypoints, descriptors = sift.compute(image, keypoints)
    return keypoints, descriptors  

# Directory containing the images
folder_path = 'extracted_frames/test/processed'

# Function to extract number from filename
def extract_number(filename):
    s = re.findall("\d+", filename)
    return int(s[0]) if s else -1

# List of image files in the folder, sorted numerically
image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))], key=extract_number)

# Iterate over pairs of images
for i in range(len(image_files) - 1):
    image1_path = os.path.join(folder_path, image_files[i])
    image2_path = os.path.join(folder_path, image_files[i + 1])

    # Load the two images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convert images to grayscale for SIFT
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Get facial landmarks
    landmarks1 = get_landmarks(gray1)
    landmarks2 = get_landmarks(gray2)

    # Return SIFT keypoints and descriptors
    keypoints1, descriptors1 = get_sift_descriptors(gray1, landmarks1)
    keypoints2, descriptors2 = get_sift_descriptors(gray2, landmarks2)

    # Find matches between the two images
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Define the maximum slope ratio for a good match
    max_slope_ratio = 0.2 # Change this value as needed

    # Apply ratio test and slope test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # David Lowe ratio
            # Calculate slope between keypoints
            pt1 = keypoints1[m.queryIdx].pt
            pt2 = keypoints2[m.trainIdx].pt
            slope = abs((pt2[1] - pt1[1]) / (pt2[0] - pt1[0]) if pt2[0] != pt1[0] else float('inf'))
            if slope < max_slope_ratio:
                good_matches.append(m)

    # Draw the matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)

    # Show the two images with the matched keypoints
    cv2.imshow(f"Matches between {image_files[i]} and {image_files[i+1]}", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
