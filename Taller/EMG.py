import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Taller.main import cargar_senal, graficar_senal, filtrar_senal, graficar_filtro, \
                        detectar_onset_offset_auto, teager_kaiser, calcular_rms , \
                        graficar_resultados_emg, recortar_segmento, graficar_teager_y_rms, \
                        graficar_zoom_contracciones

# Cargar señal ECG
emg_signal, t, fs = cargar_senal('/content/Procesamiento/Audios/Senales/musculos1.mat', tipo='mat', variable='musculos')

# Recortar la senal en el tiempo para una mejor visualizacion 
emg_recortada, t_recortado = recortar_segmento(t, emg_signal, 1000, 2500)

# Graficar
graficar_senal(t_recortado, emg_recortada, "EMG", ylabel="Amplitud")

# Filtrado pasa banda típico EM
lowcut = 20
highcut = 450
emg_filtrada, b, a = filtrar_senal(emg_recortada, fs, lowcut, highcut)

# Graficar respuesta en frecuencia y plano Z
graficar_filtro(b, a, fs)

# 1. Aplicar Teager-Kaiser a la señal EMG filtrada
tk_signal = teager_kaiser(emg_filtrada)
tk_time = t_recortado[1:-1]  # Ajustar tiempo

# 2. Calcular RMS sobre la señal TK
window_size = int(0.250 * fs)
rms = calcular_rms(tk_signal, window_size)
t_rms = tk_time[:len(rms)]

# Graficar el TK y la senal con RMS aplicado
graficar_teager_y_rms(tk_time, tk_signal, t_rms, rms)

# Detectar onsets y offsets en el RMS
onsets, offsets, ratio_usado, threshold = detectar_onset_offset_auto(rms, t_rms)
# Verificar onset y offset y ver puntos
print("Onset times:", t_rms[onsets])
print("Offset times:", t_rms[offsets])

# Graficar onset y offset con zoom para evidenciar puntos de inicio y final
graficar_zoom_contracciones(t_rms, rms, onsets, offsets, zoom_duracion=2)

# Conjunto de graficar de todos los pasos aplicados en el proceso de filtrado de la senal
graficar_resultados_emg(
    t_recortado,
    emg_recortada,
    emg_filtrada,
    tk_signal,
    rms,
    t_rms,
    onsets,
    offsets
)