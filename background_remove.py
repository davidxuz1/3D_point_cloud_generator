###################################################################################################################################################
# Clase BackgroundRemover: se utiliza para eliminar el fondo de imágenes.
# La clase permite cargar imágenes de un directorio de entrada, procesarlas utilizando el algoritmo GrabCut para la segmentación de fondo,
# y guardar las imágenes resultantes con el fondo eliminado en un directorio de salida.
# Es necesario proporcionar las rutas de las carpetas de entrada y salida al inicializar la clase.

# Dependencias necesarias:
# pip install opencv-python
# pip install numpy
###################################################################################################################################################

import cv2
import os
import numpy as np

class BackgroundRemover:
    
    def __init__(self, input_folder, output_folder):
        """
        Inicializa la clase con las rutas de las carpetas de entrada y salida.
        Crea la carpeta de salida si no existe.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def remove_background(self, image_path):
        """
        Elimina el fondo de la imagen especificada utilizando el algoritmo GrabCut.
        Retorna la imagen con el fondo eliminado.
        """
        img = cv2.imread(image_path)  # Cargar la imagen

        mask = np.zeros(img.shape[:2], np.uint8)  # Crear una máscara

        # Inicializar modelos de fondo y primer plano para GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Definir un rectángulo alrededor del objeto
        rect = (10, 10, img.shape[1]-10, img.shape[0]-10)

        # Aplicar GrabCut
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Modificar la máscara para crear una máscara binaria del primer plano
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Eliminar el fondo de la imagen
        img_no_bg = img * mask2[:, :, np.newaxis]

        return img_no_bg
    
    def process_image(self, image_name):
        """
        Procesa una imagen específica eliminando su fondo.
        Guarda la imagen procesada en la carpeta de salida.
        """
        input_path = os.path.join(self.input_folder, image_name)  # Definir ruta de entrada
        output_path = os.path.join(self.output_folder, image_name)  # Definir ruta de salida

        result_img = self.remove_background(input_path)  # Eliminar fondo

        cv2.imwrite(output_path, result_img)  # Guardar imagen procesada
        print(f"Imagen procesada guardada en {output_path}")

# Para probar la clase BackgroundRemover, descomenta las siguientes líneas y ajusta las rutas según sea necesario.
# remover = BackgroundRemover('./extracted_frames_1', './processed_frames')
# remover.process_image('frame_0.png')

