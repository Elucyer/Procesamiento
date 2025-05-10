import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def cargar_imagen(ruta):
    return cv2.imread(ruta)

def convertir_grises(imagen):
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

def convertir_binaria(imagen_gray):
    _, binaria = cv2.threshold(imagen_gray, 127, 255, cv2.THRESH_BINARY)
    return binaria

def convertir_tozero(imagen_gray):
    _, tozero = cv2.threshold(imagen_gray, 127, 255, cv2.THRESH_TOZERO)
    return tozero

def transformada_fourier(imagen_gray):
    dft = cv2.dft(np.float32(imagen_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitud = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return np.uint8(magnitud)

def transformada_wavelet(imagen_gray):
    import pywt
    coeffs2 = pywt.dwt2(imagen_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return np.uint8(LL)

def recortar_imagen(imagen, x, y, w, h):
    return imagen[y:y+h, x:x+w]

def ecualizar_histograma(imagen_gray):
    return cv2.equalizeHist(imagen_gray)

def mostrar_resultados(imagen_original, imagen_gray, binaria, tozero, fourier, wavelet):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    titulos = ["Original", "Grises", "Binaria", "ToZero", "Fourier", "Wavelet"]
    imagenes = [imagen_original, imagen_gray, binaria, tozero, fourier, wavelet]
    
    for i in range(6):
        if len(imagenes[i].shape) == 2:
            axes[i].imshow(imagenes[i], cmap='gray')
        else:
            axes[i].imshow(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2RGB))
        axes[i].set_title(titulos[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def guardar_resultados(nombre, imagen_original, imagen_gray, binaria, tozero, fourier, wavelet):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    titulos = ["Original", "Grises", "Binaria", "ToZero", "Fourier", "Wavelet"]
    imagenes = [imagen_original, imagen_gray, binaria, tozero, fourier, wavelet]
    
    for i in range(6):
        if len(imagenes[i].shape) == 2:
            axes[i].imshow(imagenes[i], cmap='gray')
        else:
            axes[i].imshow(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2RGB))
        axes[i].set_title(titulos[i])
        axes[i].axis('off')

    plt.tight_layout()
    os.makedirs("resultados", exist_ok=True)
    ruta_guardado = os.path.join("resultados", f"{nombre}.png")
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"Imagen guardada en: {ruta_guardado}")