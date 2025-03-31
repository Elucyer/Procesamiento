import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Taller.main import cargar_audios_desde_carpeta, escuchar_y_animar_audio, fft_y_espectrograma

## El audio 5 varia segun la organizacion que se tenga en este caso para nosotros sea el audio #7 el CPE.mp4

# Ruta donde estan los audios modificar segun lugar donde se guardan
carpeta_audios = "/content/Procesamiento/Audios/audio-main"

# listar los audios en la carpeta segun las extensiones abmitidas
audios = cargar_audios_desde_carpeta(carpeta_audios)

# Mostrar lista de archivos
print("\nAudios disponibles:")
for i, name in enumerate(audios.keys()):
    print(f"{i}: {name}")

# Muestra mensaje en pantalla para que se pueda escoger el audio que se desea escuchar si se aplica las recomendaciones de la linea 20 y 21 de resto ejecuta el audio por defecto 
## Si se desea que la interfaz sea mas interactiva y que el usuario pueda decidir que audio escuchar descomentar la linea 22
## y la linea  23 reemplazarla por ----> nombre_seleccionado = list(audios.keys())[index]
index = int(input("\nIngresa el número del audio que quieres usar: "))
nombre_seleccionado = list(audios.keys())[index]
audio = audios[nombre_seleccionado]
data = audio['data']
samplerate = audio['samplerate']

# grafica la senal y muestra el audio en tiempo real en el desplazamiento de la senal
escuchar_y_animar_audio(nombre_seleccionado, data, samplerate)

# Graficar Tranformada de fourier y espectograma en db
fft_y_espectrograma(data, samplerate)
