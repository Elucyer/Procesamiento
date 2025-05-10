import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Imagenes.main import (
    cargar_imagen, convertir_grises, convertir_binaria, convertir_tozero,
    transformada_fourier, transformada_wavelet, recortar_imagen,
    ecualizar_histograma, mostrar_resultados, guardar_resultados,
    obtener_histograma, encontrar_roi, mostrar_histogramas_individuales
)

# Ruta a la carpeta con las imágenes
carpeta = r'C:\Users\Usuario\OneDrive\Documents\Proyectos\Universidad\Procesamiento\Imagenes\imagenes_manolo'
if not os.path.exists(carpeta):
    raise FileNotFoundError(f"La carpeta '{carpeta}' no existe.")

imagenes = [os.path.join(carpeta, archivo) for archivo in os.listdir(carpeta) if archivo.endswith('.tiff')]

for idx, ruta in enumerate(imagenes, start=1):
    imagen = cargar_imagen(ruta)
    if imagen is None:
        print(f"No se pudo cargar la imagen: {ruta}")
        continue

    imagen_gray = convertir_grises(imagen)
    imagen_bin = convertir_binaria(imagen_gray)
    imagen_tozero = convertir_tozero(imagen_gray)
    imagen_fourier = transformada_fourier(imagen_gray)
    imagen_wavelet = transformada_wavelet(imagen_gray)

    x, y, w, h = encontrar_roi(imagen_gray)
    imagen_crop = recortar_imagen(imagen_gray, x, y, w, h)
    imagen_eq = ecualizar_histograma(imagen_crop)
    hist_crop = obtener_histograma(imagen_crop)
    hist_eq = obtener_histograma(imagen_eq)
    
    nombre_archivo = f"imagen{idx}"
    
    mostrar_resultados(imagen, imagen_gray, imagen_bin, imagen_tozero,
                   imagen_fourier, imagen_wavelet,
                   imagen_crop, imagen_eq, hist_crop, hist_eq)
    
    mostrar_histogramas_individuales(hist_crop, hist_eq)

    guardar_resultados(nombre_archivo, imagen, imagen_gray, imagen_bin, imagen_tozero,
                    imagen_fourier, imagen_wavelet, imagen_crop, imagen_eq, hist_crop, hist_eq)
