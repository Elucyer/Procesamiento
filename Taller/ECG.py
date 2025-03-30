import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Taller.main import cargar_senal, graficar_senal, fft_y_espectrograma, detectar_picos_r, calcular_frecuencia_cardiaca

# Cambia el nombre del archivo y el tipo según el caso:
# Para ECG en CSV:
# ecg_signal, t, fs = cargar_senal('ecg_data.csv', tipo='csv', fs=360)

# Para archivo de audio en mp3 o mp4:
ecg_signal, t, fs = cargar_senal('/Users/janerperez/Documents/Universidad/Procesamiento/Audios/audio-main/AIns.mp4', tipo='mp4')  # Se detecta automáticamente el fs real del archivo

# Graficar señal
graficar_senal(t, ecg_signal, "Señal ECG o Audio", ylabel="Amplitud")

# FFT y espectrograma
fft_y_espectrograma(ecg_signal, fs)

# Detección de picos R (solo aplicable si es una señal ECG válida)
peaks = detectar_picos_r(ecg_signal, fs)
hr = calcular_frecuencia_cardiaca(peaks, fs)
print(f"Frecuencia cardíaca estimada: {hr:.2f} bpm")