import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Taller.main import cargar_senal, graficar_senal, fft_y_espectrograma, calcular_frecuencia_cardíaca_pantompkin, pan_tompkins

# Leer archivo de audio en mp3 o mp4:
ecg_signal, t, fs = cargar_senal('/Users/janerperez/Documents/Universidad/Procesamiento/Audios/audio-main/AIns.mp4', tipo='mp4')

# Graficar señal
graficar_senal(t, ecg_signal, "Señal ECG o Audio", ylabel="Amplitud")

# FFT y espectrograma
fft_y_espectrograma(ecg_signal, fs)

# Llamar al algoritmo de Pan-Tompkins
peaks_pan = pan_tompkins(ecg_signal, fs)

# Calcular BPM y mostrar resultados
bpm, n_picos = calcular_frecuencia_cardíaca_pantompkin(peaks_pan, fs)
print(f"Picos R detectados: {n_picos}")
print(f"Frecuencia cardíaca estimada: {bpm:.2f} BPM")
