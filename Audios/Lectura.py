import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import librosa

# Ruta de la carpeta
audio_folder = "/Users/janerperez/Documents/Universidad/Procesamiento/Audios/audio-main"

# Extensiones de audio soportadas
extensiones_validas = ('.wav', '.flac', '.aiff', '.aif', '.mp3', '.mp4')

# Leer todos los archivos de audio
audio_files = [f for f in os.listdir(audio_folder) if f.lower().endswith(extensiones_validas)]

# Cargar audios como arrays de numpy
audios = {}
for filename in audio_files:
    path = os.path.join(audio_folder, filename)
    try:
        if filename.lower().endswith(('.mp3', '.mp4')):
            # Usamos librosa para mp3/mp4
            data, samplerate = librosa.load(path, sr=None, mono=False)  # mantiene estéreo si lo hay
            data = data.T if data.ndim == 2 else data
        else:
            # Usamos soundfile para otros formatos
            data, samplerate = sf.read(path)
        audios[filename] = {'data': data, 'samplerate': samplerate}
        print(f"Cargado: {filename}")
    except Exception as e:
        print(f"Error cargando {filename}: {e}")

# Mostrar archivos disponibles
print("\nAudios disponibles:")
for i, name in enumerate(audios.keys()):
    print(f"{i}: {name}")

# Seleccionar audio
index = int(input("\nIngresa el número del audio que quieres usar: "))
selected_name = list(audios.keys())[index]
selected_audio = audios[selected_name]
data = selected_audio['data']
samplerate = selected_audio['samplerate']

# Reproducir audio
print(f"\nReproduciendo: {selected_name}")
sd.play(data, samplerate)
sd.wait()

# Graficar forma de onda
plt.figure(figsize=(10, 4))

if data.ndim == 1:
    plt.plot(np.linspace(0, len(data) / samplerate, num=len(data)), data)
    plt.title(f"Forma de onda: {selected_name}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
else:
    for i in range(data.shape[1]):
        plt.plot(np.linspace(0, len(data) / samplerate, num=len(data)), data[:, i], label=f'Canal {i+1}')
    plt.title(f"Forma de onda (estéreo): {selected_name}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()