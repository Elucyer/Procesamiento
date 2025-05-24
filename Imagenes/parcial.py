import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, io, color, exposure, morphology, segmentation, measure, util
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.draw import line as draw_line
import pywt
import pandas as pd

# --- Carga de imagen ---
image_path = r'C:\Users\Usuario\OneDrive\Documents\Proyectos\Universidad\Procesamiento\Imagenes\imagenes_manolo\parcial\tac-de-la-cabeza1.jpg'
image = io.imread(image_path)

if image.ndim == 3:
    gray = color.rgb2gray(image)
else:
    gray = image

# --- Ajuste gamma para mejorar contraste ---
image_gamma = exposure.adjust_gamma(gray, gamma=1.3, gain=1)

# --- Umbralización con Otsu ---
thresh = filters.threshold_otsu(image_gamma)
binary = image_gamma > thresh

# --- Limpieza morfológica ---
cleaned = morphology.remove_small_objects(binary, min_size=200)
closed = morphology.binary_closing(cleaned, morphology.disk(3))

# --- Eliminar bordes tocados ---
cleared = segmentation.clear_border(closed)

# --- Etiquetar regiones ---
label_image = measure.label(cleared)
image_label_overlay = color.label2rgb(label_image, image=image, bg_label=0)

# --- Medición de propiedades ---
props = measure.regionprops_table(label_image, intensity_image=gray,
                                  properties=['label', 'area', 'mean_intensity', 'equivalent_diameter', 'solidity'])
df = pd.DataFrame(props)
df = df[df['area'] > 200]
print(df.head())

# --- Ecualización global ---
ecualizada = exposure.equalize_hist(gray)

# --- Wavelet (cA) ---
coeffs = pywt.dwt2(gray, 'haar')
cA, (cH, cV, cD) = coeffs
wavelet_img = np.uint8(np.clip(cA * 255.0 / np.max(cA), 0, 255))

# --- Ecualización CLAHE (adaptativa) ---
gray_8bit = np.uint8(gray * 255)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray_8bit)

# --- Transformada de Hough para líneas rectas ---
edges = canny(gray)
lines = probabilistic_hough_line(edges, threshold=10, line_length=30, line_gap=5)

hough_img = np.zeros_like(gray)
for p0, p1 in lines:
    rr, cc = draw_line(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
    hough_img[rr, cc] = 1

# --- Visualización en collage 3x4 ---
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
fig.suptitle('Procesamiento de Imagen TAC Craneal', fontsize=16)

# Fila 1
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title("Original")

axes[0, 1].imshow(binary, cmap='gray')
axes[0, 1].set_title("Umbralización (Otsu)")

axes[0, 2].imshow(closed, cmap='gray')
axes[0, 2].set_title("Cierre Morfológico")

axes[0, 3].imshow(image_label_overlay)
axes[0, 3].set_title("Segmentación Etiquetada")

# Fila 2
axes[1, 0].imshow(ecualizada, cmap='gray')
axes[1, 0].set_title("Ecualización Global")

axes[1, 1].imshow(wavelet_img, cmap='gray')
axes[1, 1].set_title("Wavelet (cA)")

axes[1, 2].imshow(hough_img, cmap='gray')
axes[1, 2].set_title("Transformada de Hough")

axes[1, 3].imshow(clahe_img, cmap='gray')
axes[1, 3].set_title("Ecualización CLAHE")

# Fila 3 vacía
for i in range(4):
    axes[2, i].axis('off')

# Quitar ejes
for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
