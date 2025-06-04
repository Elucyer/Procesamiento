import cv2
import numpy as np
from skimage import color, morphology, segmentation, measure
import matplotlib.pyplot as plt
import pywt

def cargar_imagen(ruta):
    return cv2.imread(ruta)

def procesar_imagen(imagen_rgb):
    # 1. Convertir imagen RGB a espacio de color HSV
    # HSV permite segmentar colores de forma más robusta que RGB
    imagen_hsv = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2HSV)

    # 2. Crear máscara para identificar regiones púrpuras (parásitos)
    # Definimos un rango en HSV que corresponde al color de los parásitos
    lower_purple = np.array([120, 30, 40])
    upper_purple = np.array([170, 255, 255])
    mascara = cv2.inRange(imagen_hsv, lower_purple, upper_purple)

    # 3. Limpieza morfológica de la máscara
    # - Elimina objetos pequeños que podrían ser ruido
    # - Aplica cierre morfológico para rellenar huecos dentro de regiones
    mascara_clean = morphology.remove_small_objects(mascara > 0, min_size=100)
    mascara_closed = morphology.binary_closing(mascara_clean, morphology.disk(2))

    # 4. Segmentación y etiquetado de regiones conectadas
    # - Elimina regiones tocando los bordes
    # - Etiqueta componentes conectados y asigna color a cada uno
    cleared = segmentation.clear_border(mascara_closed)
    label_image = measure.label(cleared)
    segmentada = color.label2rgb(label_image, image=imagen_rgb, bg_label=0)

    # 5. Mejora de contraste en canal de brillo (V) usando CLAHE
    # CLAHE (ecualización adaptativa) mejora detalles locales sin sobresaturar
    imagen_v = imagen_hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(imagen_v)

    # 6. Transformada Wavelet sobre imagen en escala de grises
    # Extraemos la aproximación (baja frecuencia) para resaltar estructuras gruesas
    gray = color.rgb2gray(imagen_rgb)
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (_, _, _) = coeffs
    wavelet_img = np.uint8(np.clip(cA * 255.0 / np.max(cA), 0, 255))

    # Devuelve todas las etapas del procesamiento
    return imagen_rgb, mascara * 255, segmentada, clahe_img, wavelet_img

def mostrar_comparacion(lista_parasitadas, lista_no_parasitadas):
    columnas = ["Original", "Máscara Color", "Segmentación", "CLAHE", "Wavelet"]
    total = min(len(lista_parasitadas), len(lista_no_parasitadas), 10)  # Limita la comparación a 10 pares

    for i in range(total):
        nombre_p, imagen_p = lista_parasitadas[i]
        nombre_n, imagen_n = lista_no_parasitadas[i]

        # Crear figura con 2 filas (parasitada y no parasitada) y 5 columnas (etapas de procesamiento)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"Comparación: {nombre_p} vs {nombre_n}", fontsize=16)

        # Procesar ambas imágenes
        datos_parasitada = procesar_imagen(imagen_p)
        datos_no_parasitada = procesar_imagen(imagen_n)

        # Mostrar resultados paso a paso
        for j, titulo in enumerate(columnas):
            # Fila superior: imagen parasitada
            axes[0, j].imshow(datos_parasitada[j], cmap='gray' if j in [1, 3, 4] else None)
            axes[0, j].set_title(f"{titulo} (Parasitada)")
            axes[0, j].axis('off')

            # Fila inferior: imagen no parasitada
            axes[1, j].imshow(datos_no_parasitada[j], cmap='gray' if j in [1, 3, 4] else None)
            axes[1, j].set_title(f"{titulo} (No Parasitada)")
            axes[1, j].axis('off')

        # Ajustar espaciado para evitar solapamiento
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

