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
    ruta_dataset = descargar_dataset()
    ruta_parasitadas = os.path.join(ruta_dataset, r"cell_images\Parasitized")
    ruta_no_parasitadas = os.path.join(ruta_dataset, r"cell_images\Uninfected")

        # Obtener nombres de archivos PNG
    nombres_parasitadas = sorted([f for f in os.listdir(ruta_parasitadas) if f.endswith(".png")])[:10]
    nombres_no_parasitadas = sorted([f for f in os.listdir(ruta_no_parasitadas) if f.endswith(".png")])[:10]

    # Cargar imágenes junto con sus nombres
    imagenes_parasitadas = [(f, cargar_imagen(os.path.join(ruta_parasitadas, f))) for f in nombres_parasitadas]
    imagenes_no_parasitadas = [(f, cargar_imagen(os.path.join(ruta_no_parasitadas, f))) for f in nombres_no_parasitadas]


    print("Comparando imágenes de células parasitadas vs no parasitadas...")
    mostrar_comparacion(imagenes_parasitadas, imagenes_no_parasitadas)


if __name__ == "__main__":
    main()
