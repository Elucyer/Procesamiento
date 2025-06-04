import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from kaggle.api.kaggle_api_extended import KaggleApi
from main import cargar_imagen, mostrar_comparacion

def descargar_dataset():
    ruta_destino = r"C:\Users\Usuario\OneDrive\Documents\Proyectos\Universidad\Procesamiento\Trabajo_Final_Celulas_Malaria\celulas_malaria"
    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("iarunava/cell-images-for-detecting-malaria", path=ruta_destino, unzip=True)
    return ruta_destino


def main():
    # Paso 1: Descargar y descomprimir el dataset desde Kaggle si no existe
    ruta_dataset = descargar_dataset()

    # Paso 2: Establecer las rutas a las carpetas con imágenes parasitadas y no parasitadas
    ruta_parasitadas = os.path.join(ruta_dataset, r"Parasitized")
    ruta_no_parasitadas = os.path.join(ruta_dataset, r"Uninfected")

    # Paso 3: Seleccionamos las primeras 10 imágenes PNG de cada clase
    nombres_parasitadas = sorted([f for f in os.listdir(ruta_parasitadas) if f.endswith(".png")])[:10]
    nombres_no_parasitadas = sorted([f for f in os.listdir(ruta_no_parasitadas) if f.endswith(".png")])[:10]

    # Paso 4: Cargar imágenes en memoria junto con sus nombres
    imagenes_parasitadas = [(f, cargar_imagen(os.path.join(ruta_parasitadas, f))) for f in nombres_parasitadas]
    imagenes_no_parasitadas = [(f, cargar_imagen(os.path.join(ruta_no_parasitadas, f))) for f in nombres_no_parasitadas]

    # Paso 5: Comparar visualmente las imágenes de cada clase (sanas vs. enfermas)
    print("Comparando imágenes de células parasitadas vs no parasitadas...")
    mostrar_comparacion(imagenes_parasitadas, imagenes_no_parasitadas)



if __name__ == "__main__":
    main()
