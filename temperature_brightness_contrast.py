import cv2
import os

###################################################################################################################################################
# pip install opencv-contrib-python
# pip install opencv-python
# pip install numpy
# pip install os-sys

# Este script es parte de un flujo de trabajo para la reconstrucción de imágenes y visualización 3D de rostros.
# La clase ImageProcessor procesa imágenes para ajustar el balance de blancos y mejorar el contraste utilizando CLAHE.
# Las imágenes se almacenarán en un directorio de salida después del procesamiento.
###################################################################################################################################################

class ImageProcessor:
    
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def auto_white_balance(self, img):
        """
        Aplica balance automático de blancos a la imagen proporcionada.
        Retorna la imagen con el balance de blancos ajustado.
        """
        # Crear un objeto para el balance de blancos y ajustar la imagen
        wb = cv2.xphoto.createSimpleWB()
        corrected_img = wb.balanceWhite(img)
        return corrected_img
    
    def apply_clahe_color(self, img):
        """
        Aplica el algoritmo CLAHE (Contrast Limited Adaptive Histogram Equalization) a una imagen en color.
        Mejora el contraste de la imagen en el espacio de color LAB.
        Retorna la imagen con CLAHE aplicado.
        """
        # Convertir la imagen de BGR a LAB
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        
        # Separar los canales LAB
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        
        # Crear el objeto CLAHE y aplicarlo al canal L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        
        # Combinar los canales de nuevo
        img_clahe = cv2.merge((cl, a_channel, b_channel))
        
        # Convertir la imagen de LAB a BGR
        img_clahe_bgr = cv2.cvtColor(img_clahe, cv2.COLOR_Lab2BGR)
        
        return img_clahe_bgr
    
    """
    def resize_image(self, img, target_width=512, target_height=512):
        # Mantener la relación de aspecto
        aspect_ratio = img.shape[1] / img.shape[0]
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

        if new_height > target_height:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)

        resized_img = cv2.resize(img, (new_width, new_height))

        # Rellenar con negro para alcanzar el tamaño deseado
        top_padding = (target_height - new_height) // 2
        bottom_padding = target_height - new_height - top_padding
        left_padding = (target_width - new_width) // 2
        right_padding = target_width - new_width - left_padding

        padded_img = cv2.copyMakeBorder(resized_img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[255,255,255])

        return padded_img
    """
    
    def process_images(self):
        # Cargar todas las imágenes PNG de la carpeta de entrada
        image_paths = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.endswith('.png')]
        
        for image_path in image_paths:
            # Leer la imagen
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            # Aplicar CLAHE
            img_clahe = self.apply_clahe_color(img)
            
            # Ajustar balance de blancos
            adjusted_img = self.auto_white_balance(img_clahe)
            
            # Redimensionar la imagen
            #resized_img = self.resize_image(adjusted_img)

            # Guardar la imagen procesada en la carpeta de salida
            save_path = os.path.join(self.output_folder, os.path.basename(image_path))
            cv2.imwrite(save_path, adjusted_img)
            print(f"Imagen {image_path} procesada y guardada en {save_path}")


# para probar el código: png en la carpeta /extracted_frames/test
#processor = ImageProcessor('./extracted_frames/test', './processed_frames')
#processor.process_images()