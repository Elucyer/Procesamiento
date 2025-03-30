import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Taller.main import fft_y_espectrograma, graficar_senal, detectar_picos_r, calcular_frecuencia_cardiaca, cargar_senal

# Cargar señal ECG
ecg_signal, t, fs = cargar_senal('ecg_data.csv', tipo='csv', fs=360)

# Graficar
graficar_senal(t, ecg_signal, "ECG")

# FFT y espectrograma
fft_y_espectrograma(ecg_signal, fs)

# Detección R y HR
peaks = detectar_picos_r(ecg_signal, fs)
hr = calcular_frecuencia_cardiaca(peaks, fs)
print(f"Frecuencia cardíaca: {hr:.2f} bpm")