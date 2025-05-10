import os
from Procesamiento.Imagenes.main import (
    cargar_imagen, convertir_grises, convertir_binaria, convertir_tozero,
    transformada_fourier, transformada_wavelet, recortar_imagen,
    ecualizar_histograma, mostrar_resultados, guardar_resultados
)

# Ruta a la carpeta con las imágenes
carpeta = 'imagenes_termograficas'
imagenes = [os.path.join(carpeta, archivo) for archivo in os.listdir(carpeta) if archivo.endswith('.tiff')]

# Parámetros para recorte (ajustar según ROI de tus imágenes)
x, y, w, h = 50, 50, 200, 200

for ruta in imagenes:
    imagen = cargar_imagen(ruta)
    imagen_gray = convertir_grises(imagen)
    imagen_bin = convertir_binaria(imagen_gray)
    imagen_tozero = convertir_tozero(imagen_gray)
    imagen_fourier = transformada_fourier(imagen_gray)
    imagen_wavelet = transformada_wavelet(imagen_gray)
    
    imagen_crop = recortar_imagen(imagen_gray, x, y, w, h)
    imagen_eq = ecualizar_histograma(imagen_crop)
    
    nombre_archivo = f"imagen{ruta+1}"
    mostrar_resultados(imagen, imagen_gray, imagen_bin, imagen_tozero, imagen_fourier, imagen_wavelet)
    guardar_resultados(nombre_archivo, imagen, imagen_gray, imagen_bin, imagen_tozero, imagen_fourier, imagen_wavelet)
