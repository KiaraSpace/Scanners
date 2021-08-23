# DATATHON 2021 INNOVA ENTEL
-----------------------------------------------------------------------------------------------------------------------------------------------
## Información del grupo
1. Nombre del grupo: **Scanners**
2. Nombre de los integrantes:
    1. Dari Joshua Acuña Quiñones
    2. Kiara Micaela Rodriguez Bautista
    3. Jürgen Anders Guerra Ramos
-----------------------------------------------------------------------------------------------------------------------------------------------
## Instalación
1. Instalar [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/doc/doc_en/installation_en.md).
2. Dentro de la carpeta clonada de PaddleOCR, clonar este repositorio. 

   ``` 
   git submodule add https://github.com/KiaraSpace/Scanners.git 
   ```
3. Instalar paquetes extras necesarios: `opencv-python` y `imutils`.
-----------------------------------------------------------------------------------------------------------------------------------------------
## Ejecución
1. Ejecutar primero `preprocessing.py`.

   Este script primero detecta el cuadro de interés de cada hoja para cada imagen dentro de la carpeta `image_test`. Una vez detectado, se realiza la transformación de perspectiva de dicho cuadro. La fecha dentro del cuadro está siempre en la misma región, por lo que podemos definir una región preestablecida con respecto al cuadro y garantizar que la fecha se encuentre dentro de ella. Así, a partir del cuadro "escaneado", recortamos la fecha. Todos los recortes se guardan en la carpeta `cropped`.
   
2. Importar la carpeta `cropped` en este [notebook](https://colab.research.google.com/drive/1wnR7jYNdVnemmToW6p37uRM7Xu3r31Z4?usp=sharing) y ejecutar el código. Descargar la carpeta que se genere después de la ejecución del código.

    asdjaksdadsakdsa

3. Extraer contenidos de la carpeta descargada en la carpeta `image_test`.
4. Ejecutar el script principal `main.py`.

   En este script se obtiene el status de las firmas por detección de objetos, usando tiny-yolov4, sobre las imágenes "crudas" de `image_test`; y se obtienen las fechas por medio de reconocimiento de texto usando el modelo PaddleOCR. Una vez recolectada toda esta información, los datos son escritos en un archivo de texto `.txt`, el output deseado, según el formato indicado por las bases. Los pesos del modelo de detección y los archivos de configuración respectivos se encuentran en la carpeta `model`.
-----------------------------------------------------------------------------------------------------------------------------------------------
## Referencias
1. Modelo de reconocimiento de texto: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/doc/doc_en/installation_en.md).
2. Modelo de restauración de imágenes: [Bringing Old Photo Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life).
3. Modelo de detección de objetos: [tiny-yolov4](https://github.com/AlexeyAB/darknet).
