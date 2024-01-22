# 3D Face Model Generator

Este proyecto contiene varios scripts de Python para la reconstrucción y visualización en 3D de rostros a partir de imágenes procesadas. Se utilizan diferentes técnicas y herramientas, incluyendo MediaPipe y métodos tradicionales de Visión por Computador.



## Estructura del Proyecto

Los siguientes scripts son parte del proyecto y cada uno tiene un propósito específico:

1. **PointCloudGenerator3D.py**: Este es el script principal para generar la nube de puntos 3D. Utiliza otros scripts auxiliares para procesar las imágenes y reconstruir la estructura 3D.

2. **face_reconstruction_mediapipe.py**: Este script utiliza MediaPipe para obtener una nube de puntos de la cara a partir de las imágenes procesadas. Se utiliza como alternativa para realizar la reconstrucción 3D usando las capacidades de MediaPipe.

3. **matching_mediapipe.py**: Este script se emplea para comprobar la eficacia del matching entre dos frames. 

4. **Otros Archivos Auxiliares**: Los otros archivos en el proyecto son scripts auxiliares que apoyan las funcionalidades del `PointCloudGenerator3D.py`. Cada uno de estos archivos contiene comentarios detallados sobre su funcionalidad específica.




## Instrucciones de Uso

Para utilizar los scripts de este proyecto, sigue estos pasos:

1. Asegurarse de tener instaladas todas las dependencias necesarias.

2. Ejecutar `PointCloudGenerator3D.py` para realizar la reconstrucción de la nube de puntos 3D.

   Alternativamente, se puede ejecutar `face_reconstruction_mediapipe.py` para obtener una nube de puntos usando MediaPipe (se debe tener los frames en el directorio)

   Se puede ejecutar `matching_mediapipe.py` para comprobar la eficacia del matching entre dos frames.

Cada script generará salidas específicas que podrás visualizar o utilizar para análisis posteriores.



## Requisitos

- Python 3.11
- Bibliotecas: OpenCV, MediaPipe, NumPy, PyVista, Matplotlib, PlyFile.



## Cómo Funciona

El proceso de reconstrucción de nubes de puntos 3D en este proyecto implica varios pasos clave:

1. **Preprocesamiento del Video: Extracción de frames**: 
   - Se comienza con un video ubicado en la carpeta `raw_video` llamado `test.mp4`.
   - El preprocesamiento implica primero la extracción de frames del video, lo cual se realiza con la clase `FrameExtractor` del archivo `extract_frames.py`. La tasa de extracción de frames se especifica según la necesidad.

2. **Preprocesamiento del Video: Ajuste de Blancos, Temperatura y Contraste**:
   - Se utiliza la clase `ImageProcessor` del archivo `temperature_brightness_contrast.py`.
   - En esta etapa se aplica el algoritmo de CLAHE para mejorar el contraste de la imagen y la temperatura de la imagen. También se ajusta el balance de blancos.

3. **Preprocesamiento del Video: Eliminación de Fondo**:
   - Se emplea la clase `BackgroundRemover` del archivo `background_remove.py`.
   - Esta clase utiliza el algoritmo GrabCut para quitar el fondo de las imágenes y mantener únicamente el rostro de la persona.

4. **Reconstrucción de la Nube de Puntos 3D**:
   - Las imágenes preprocesadas se procesan con la clase `PointCloudReconstructor` del archivo `face_reconstruction.py`.
   - Se utiliza la detección de puntos faciales de MediaPipe y los descriptores SIFT a partir de los landmarks.
   - Se realiza el matching entre los descriptores de dos imágenes utilizando el algoritmo BFMatcher.
   - Se continúa con la reconstrucción de la posición tridimensional de los puntos emparejados usando la matriz esencial 'E', aplicando el método RANSAC para robustez ante outliers.
   - A través de la matriz esencial, se recupera la rotación relativa 'R' y el vector de traslación 't' entre dos vistas de la cámara, determinando el movimiento y la orientación de la cámara entre dos frames.
   - Se crean las matrices de proyección para ambas cámaras, asumiendo que la primera cámara está en la posición de origen.
   - La matriz de proyección de la primera cámara es la matriz intrínseca 'K' multiplicada por una matriz identidad y un vector de ceros. Para la segunda cámara, la matriz de proyección combina la rotación 'R', la traslación 't' y la matriz intrínseca 'K'.
   - Finalmente, se realiza la triangulación de puntos para cada par de puntos correspondientes en las dos imágenes. Se calcula su posición en el espacio 3D usando las matrices de proyección de ambas cámaras y las coordenadas de los puntos correspondientes en ambas imágenes.
   - Se obtienen puntos en coordenadas homogéneas 4D, que luego se convierten a coordenadas 3D normales. El resultado es una nube de puntos 3D, donde cada punto representa la posición estimada en el espacio tridimensional de un punto correspondiente en las imágenes.



## Requisitos para la Calibración de la Cámara

La calibración precisa de la cámara es esencial para la reconstrucción efectiva de nubes de puntos 3D. Este proceso implica varios pasos técnicos, incluyendo el uso de algoritmos específicos para la detección de patrones de tablero de ajedrez:

1. **Obtener Imágenes del Tablero de Ajedrez**:
   - Se necesitan múltiples imágenes de un tablero de ajedrez capturadas desde diferentes ángulos.
   - Estas imágenes proporcionan puntos de referencia necesarios para la calibración.

2. **Detección de Patrones de Tablero de Ajedrez**:
   - Se emplea el algoritmo `cv.findChessboardCorners` de OpenCV, que es capaz de detectar las esquinas del tablero de ajedrez en las imágenes.
   - Este algoritmo busca patrones específicos de cuadros blancos y negros y localiza las intersecciones de las esquinas de estos cuadros.

3. **Refinamiento de las Esquinas Detectadas**:
   - Una vez detectadas las esquinas, se utiliza `cv.cornerSubPix` para aumentar la precisión de su localización.
   - Este paso es crucial para mejorar la exactitud de la calibración, ajustando las coordenadas de las esquinas al subpixel más cercano.

4. **Cálculo de la Matriz Intrínseca y Coeficientes de Distorsión**:
   - Con las esquinas del tablero precisamente localizadas, se calculan los parámetros de calibración de la cámara.
   - Esto incluye la matriz intrínseca, que describe las propiedades ópticas y geométricas de la cámara, y los coeficientes de distorsión, que corrigen las aberraciones de la lente.

5. **Corrección de la Distorsión de la Imagen**:
   - Utilizando los parámetros calculados, se aplican técnicas de corrección de distorsión en las imágenes.
   - Esto se realiza a través de `cv.undistort` y `cv.initUndistortRectifyMap`, ajustando las imágenes según la matriz intrínseca y los coeficientes de distorsión.
   - No es necesario para el `PointCloudGenerator3D.py`, únicamente se necesita la matriz intrínseca K.



## Contribuciones

Las contribuciones al proyecto son bienvenidas. Si deseas mejorar los scripts o añadir nuevas funcionalidades, no dudes en crear un fork del repositorio y enviar tus pull requests.



## Licencia

Este proyecto está licenciado bajo el equipo Visage de la Universidad Politécnica de Madrid.
