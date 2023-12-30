import numpy as np
import cv2 as cv
import os

###################################################################################################################################################
# Este script utiliza OpenCV para la calibración de cámaras mediante el análisis de imágenes de un tablero de ajedrez.
# El proceso implica encontrar y refinar las esquinas del tablero de ajedrez en las imágenes para establecer correspondencias entre puntos 3D 
# en el espacio real y puntos 2D en el plano de la imagen.
# Se utilizan `cv.findChessboardCorners` y `cv.cornerSubPix` para detectar y optimizar la precisión de las esquinas del tablero en cada imagen.
# Los puntos del objeto (3D) y los puntos de la imagen (2D) recopilados se usan para calcular la calibración de la cámara con `cv.calibrateCamera`, 
# generando la matriz intrínseca de la cámara, coeficientes de distorsión, vectores de rotación y traslación.
# La calibración permite corregir la distorsión de la imagen y mejorar la precisión de aplicaciones de visión por computador. Se incluyen métodos 
# para la corrección de la distorsión y el remapeo de la imagen usando `cv.undistort` y `cv.initUndistortRectifyMap`.
# Además, se calcula el error de reproyección para evaluar la calidad de la calibración.

# Requisitos: 
# - Imágenes de un tablero de ajedrez almacenadas en el directorio actual con extensión '.png'.
# - Dimensiones conocidas del tablero de ajedrez.
# - Librerías Python: numpy, cv2 (OpenCV), os.
# pip install numpy opencv-python

# Entrada: Imágenes del tablero de ajedrez para calibración de la cámara.
# Salida: Parámetros de calibración de la cámara (matriz intrínseca K), imágenes corregidas por distorsión y cálculo del error de reproyección.
###################################################################################################################################################

# Configuración del tamaño del tablero de ajedrez y del tamaño del marco
chessboardSize = (7,7)
frameSize = (480, 848) # (720, 1280)

# Criterios de terminación para la optimización
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparación de puntos del objeto, como (0,0,0), (1,0,0), (2,0,0) ..., (6,6,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

# Arrays para almacenar puntos del objeto y puntos de la imagen de todas las imágenes
objPoints = [] # Puntos 3D en el espacio real
imgPoints = [] # Puntos 2D en el plano de la imagen

# Obtención de la lista de nombres de archivos de imagen
images = [img for img in os.listdir('.') if img.endswith('.png')]
print("Imágenes encontradas: ", images)

for image in images:
    print("Procesando imagen:", image)
    img = cv.imread(image)
    if img is None:
        print("Error al cargar la imagen:", image)
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Encontrar las esquinas del tablero de ajedrez
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    if ret:
        print("Esquinas del tablero de ajedrez encontradas!")
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners)
        # Dibujar y mostrar las esquinas
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)
    else:
        print("Esquinas del tablero de ajedrez no encontradas en la imagen:", image)

cv.destroyAllWindows()

# Verificar si hay suficientes puntos para la calibración
if len(objPoints) > 0:
    # Calibración de la cámara
    print("Calibrando cámara...")
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

    print("Cámara calibrada: ", ret)
    print("\nMatriz de la cámara:\n", cameraMatrix)
    print("\nParámetros de distorsión:\n", dist)
    print("\nVectores de rotación:\n", rvecs)
    print("\nVectores de traslación:\n", tvecs)

    # Corrección de la distorsión
    img = cv.imread('cali5.png')
    if img is not None:
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

        # Desdistorsionar
        dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

        # Recortar la imagen
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite('caliResult1.png', dst)

        # Desdistorsionar con remapeo
        mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        # Recortar la imagen
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite('caliResult2.png', dst)

        # Cálculo del error de reproyección
        mean_error = 0

        for i in range(len(objPoints)):
            imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
            error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints2)
            mean_error += error

        print("\nError total: {}".format(mean_error/len(objPoints)))
    else:
        print("Error al cargar 'cali5.png' para la desdistorsión.")
else:
    print("No hay suficientes puntos para calibrar. Se omite la calibración.")
print("\n\n\n")
