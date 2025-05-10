import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pywt

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

def transformada_fourier(imagen_bgr):
    canales = cv2.split(imagen_bgr)  # Separar B, G, R
    fourier_canales = []

    for canal in canales:
        # DFT
        dft = cv2.dft(np.float32(canal), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitud = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitud_log = np.log1p(magnitud)
        magnitud_norm = cv2.normalize(magnitud_log, None, 0, 255, cv2.NORM_MINMAX)
        fourier_canales.append(np.uint8(magnitud_norm))

    # Combinar canales de nuevo en una imagen BGR
    return cv2.merge(fourier_canales)

def reconstruccion_fourier_color(imagen_bgr):
    canales = cv2.split(imagen_bgr)  # Separar B, G, R
    reconstruidas = []

    for canal in canales:
        # DFT directa
        dft = cv2.dft(np.float32(canal), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Volver a centrar las frecuencias (shift inverso)
        f_ishift = np.fft.ifftshift(dft_shift)

        # Transformada inversa
        img_back = cv2.idft(f_ishift)
        img_back_mag = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalizar para visualización
        img_back_norm = cv2.normalize(img_back_mag, None, 0, 255, cv2.NORM_MINMAX)
        reconstruidas.append(np.uint8(img_back_norm))

    # Combinar canales nuevamente
    return cv2.merge(reconstruidas)


def transformada_wavelet(imagen_gray):
    coeffs2 = pywt.dwt2(imagen_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return np.uint8(LL)

def recortar_imagen(imagen, x, y, w, h):
    return imagen[y:y+h, x:x+w]

def ecualizar_histograma(imagen_gray):
    return cv2.equalizeHist(imagen_gray)

def obtener_histograma(imagen_gray):
    hist = cv2.calcHist([imagen_gray], [0], None, [256], [0, 256])
    return hist

def mostrar_histograma(hist, titulo):
    plt.figure()
    plt.title(titulo)
    plt.xlabel('Intensidad de pixel')
    plt.ylabel('Frecuencia')
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.grid()
    plt.show()

def mostrar_resultados(imagen_original, imagen_gray, binaria, tozero,
                       fourier, wavelet, imagen_crop, imagen_eq,
                       hist_crop, hist_eq):

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 4)

    titulos = ["Original", "Grises", "Binaria", "ToZero",
               "Tranformada Fourier", "Tranformada Wavelet", "Imagen Recortada", "Ecualizada"]
    imagenes = [imagen_original, imagen_gray, binaria, tozero,
                fourier, wavelet, imagen_crop, imagen_eq]

    # Mostrar las 8 imágenes en una grilla 2x4
    for i in range(8):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        if len(imagenes[i].shape) == 2:
            ax.imshow(imagenes[i], cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2RGB))
        ax.set_title(titulos[i])
        ax.axis('off')

    # Subplot grande para los histogramas (fila 3, columnas completas)
    ax_hist = fig.add_subplot(gs[2, :])  # Fila 3, columnas 0 a 3
    ax_hist.plot(hist_crop, label='Original', color='blue')
    ax_hist.plot(hist_eq, label='Ecualizado', color='red')
    ax_hist.set_title("Histogramas de Intensidad")
    ax_hist.set_xlim([0, 256])
    ax_hist.set_xlabel("Intensidad de píxel")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.legend()
    ax_hist.grid()

    plt.tight_layout()
    plt.show()

def guardar_resultados(nombre, imagen_original, imagen_gray, binaria, tozero,
                       fourier, wavelet, imagen_crop, imagen_eq, hist_crop, hist_eq):
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 4)  # 3 filas, 4 columnas

    titulos = ["Original", "Grises", "Binaria", "ToZero",
               "Tranformada Fourier", "Tranformada Wavelet", "Imagen Recortada", "Ecualizada"]
    imagenes = [imagen_original, imagen_gray, binaria, tozero,
                fourier, wavelet, imagen_crop, imagen_eq]

    # Mostrar las 8 imágenes
    for i in range(8):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        if len(imagenes[i].shape) == 2:
            ax.imshow(imagenes[i], cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2RGB))
        ax.set_title(titulos[i])
        ax.axis('off')

    # Subplot grande para los histogramas (fila 3, columnas completas)
    ax_hist = fig.add_subplot(gs[2, :])  # Fila 3, columnas 0 a 3
    ax_hist.plot(hist_crop, label='Original', color='blue')
    ax_hist.plot(hist_eq, label='Ecualizado', color='red')
    ax_hist.set_title("Histogramas de Intensidad")
    ax_hist.set_xlim([0, 256])
    ax_hist.set_xlabel("Intensidad de píxel")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.legend()
    ax_hist.grid()

    # Guardar la figura
    plt.tight_layout()
    os.makedirs("resultados", exist_ok=True)
    ruta_guardado = os.path.join("resultados", f"{nombre}.png")
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"Imagen guardada en: {ruta_guardado}")


def encontrar_roi(imagen_gray):
    # Detección de bordes para obtener contornos
    edges = cv2.Canny(imagen_gray, 100, 200)
    
    # Buscar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Si no hay contornos, usar una región por defecto
        h, w = imagen_gray.shape
        return 0, 0, w // 2, h // 2

    # Escoger el contorno con mayor área
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Redimensionar suavemente si es muy pequeño
    if w < 50 or h < 50:
        x, y = max(0, x - 10), max(0, y - 10)
        w, h = min(imagen_gray.shape[1] - x, w + 20), min(imagen_gray.shape[0] - y, h + 20)
    
    return x, y, w, h

def mostrar_histogramas_individuales(hist_crop, hist_eq):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma del Original
    axes[0].plot(hist_crop, color='blue')
    axes[0].set_title("Histograma Original")
    axes[0].set_xlabel("Intensidad de píxel")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_xlim([0, 256])
    axes[0].grid()

    # Histograma ecualizado
    axes[1].plot(hist_eq, color='red')
    axes[1].set_title("Histograma Ecualizado")
    axes[1].set_xlabel("Intensidad de píxel")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_xlim([0, 256])
    axes[1].grid()

    plt.tight_layout()
    plt.show()
