import cv2
import numpy as np
from skimage import color, morphology, segmentation, measure
import matplotlib.pyplot as plt
import pywt

def cargar_imagen(ruta):
    return cv2.imread(ruta)

def procesar_imagen(imagen_rgb):
    # 1. Convertir a HSV
    imagen_hsv = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2HSV)

    # 2. Crear máscara para el parásito (zona púrpura)
    lower_purple = np.array([120, 30, 40])
    upper_purple = np.array([170, 255, 255])
    mascara = cv2.inRange(imagen_hsv, lower_purple, upper_purple)

    # 3. Limpieza morfológica
    mascara_clean = morphology.remove_small_objects(mascara > 0, min_size=100)
    mascara_closed = morphology.binary_closing(mascara_clean, morphology.disk(2))

    # 4. Segmentación etiquetada
    cleared = segmentation.clear_border(mascara_closed)
    label_image = measure.label(cleared)
    segmentada = color.label2rgb(label_image, image=imagen_rgb, bg_label=0)

    # 5. CLAHE en canal V
    imagen_v = imagen_hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(imagen_v)

    # 6. Wavelet (gray)
    gray = color.rgb2gray(imagen_rgb)
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (_, _, _) = coeffs
    wavelet_img = np.uint8(np.clip(cA * 255.0 / np.max(cA), 0, 255))

    return imagen_rgb, mascara * 255, segmentada, clahe_img, wavelet_img

def mostrar_comparacion(lista_parasitadas, lista_no_parasitadas):
    columnas = ["Original", "Máscara Color", "Segmentación", "CLAHE", "Wavelet"]
    total = min(len(lista_parasitadas), len(lista_no_parasitadas), 10)

    for i in range(total):
        nombre_p, imagen_p = lista_parasitadas[i]
        nombre_n, imagen_n = lista_no_parasitadas[i]

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"Comparación: {nombre_p} vs {nombre_n}", fontsize=16)

        datos_parasitada = procesar_imagen(imagen_p)
        datos_no_parasitada = procesar_imagen(imagen_n)

        for j, titulo in enumerate(["Original", "Máscara Color", "Segmentación", "CLAHE", "Wavelet"]):
            axes[0, j].imshow(datos_parasitada[j], cmap='gray' if j in [1, 3, 4] else None)
            axes[0, j].set_title(f"{titulo} (Parasitada)")
            axes[0, j].axis('off')

            axes[1, j].imshow(datos_no_parasitada[j], cmap='gray' if j in [1, 3, 4] else None)
            axes[1, j].set_title(f"{titulo} (No Parasitada)")
            axes[1, j].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

